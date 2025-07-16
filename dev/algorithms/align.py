import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import scipy.spatial
import scipy.optimize
from scipy.spatial.transform import Rotation as R
from shapely.geometry import MultiPoint, Polygon
from tqdm import tqdm
from numba import jit


def align(method, points, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    last_method = basic

    if method == "none":
        return points, 0, dict()
    elif method == "basic":
        return basic(points)
    elif method == "minimal":
        return minimum_volume_bounding_box(points)
    elif method == "approximate":
        return compute_optimal_obb(points, **parameters)
    elif method == "approximate_jit":
        return compute_optimal_obb_jit(points, **parameters)
    else:
        return last_method(points)

def basic(points):
    # Convert the points to an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Compute the oriented bounding box (MVB)
    obb = point_cloud.get_oriented_bounding_box()

    # Get the rotation matrix and the center of the bounding box
    R = obb.R
    center = obb.center

    # Align the points by rotating them according to the bounding box's rotation matrix
    aligned_points = np.dot(points - center, R.T)

    # Compute the aligned axis-aligned bounding box
    min_bound = np.min(aligned_points, axis=0)
    max_bound = np.max(aligned_points, axis=0)
    aabb_volume = np.prod(max_bound - min_bound)

    return aligned_points, aabb_volume, dict()

def compute_orthonormal_basis(u):
    """
    Given a unit vector u (3,), returns two unit vectors v and w so that
    (v, w, u) is an orthonormal basis.
    """
    # Choose a vector that is not parallel to u
    if abs(u[0]) < 0.9:
        a = np.array([1, 0, 0])
    else:
        a = np.array([0, 1, 0])
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return v, w

def min_area_rectangle_2d(points2d):
    """
    Given a set of 2D points (Nx2 array) returns the coordinates of the
    minimum-area bounding rectangle (as a (4,2) array of vertices) and its area.
    Uses shapely's minimum_rotated_rectangle.
    """
    mp = MultiPoint(points2d)
    rect = mp.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)
    # The polygon is closed (first point == last point); remove duplicate.
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    area = Polygon(coords).area
    return coords, area

def get_candidate_normals(points):
    """
    From the convex hull of points, extract a list of candidate face normals.
    (Duplicates are filtered out up to a tolerance.)
    """
    hull = ConvexHull(points)
    candidate_normals = []
    tol = 1e-6
    for eq in hull.equations:  # each eq is (a, b, c, d) for plane: a*x+b*y+c*z+d=0
        normal = eq[:3]
        norm = np.linalg.norm(normal)
        if norm < tol:
            continue
        normal = normal / norm
        # Check if a similar normal is already present (account for sign)
        add = True
        for n in candidate_normals:
            if np.allclose(normal, n, atol=tol) or np.allclose(normal, -n, atol=tol):
                add = False
                break
        if add:
            candidate_normals.append(normal)
    return candidate_normals

def minimum_volume_bounding_box(points):
    """
    Given an (N,3) array of points, computes the minimum volume bounding box.
    
    Returns a dictionary with keys:
      'volume'   : minimal volume
      'center'   : 3D center of the box
      'rotation' : (3,3) rotation matrix whose columns are the box axes.
      'extents'  : lengths along each box axis (i.e. dimensions)
      'corners'  : (8,3) array with the coordinates of the box vertices.
    """
    points = np.asarray(points)
    if points.shape[1] != 3:
        raise ValueError("Input points must be a (N,3) array.")

    all_points = points.copy()

    hull = scipy.spatial.ConvexHull(all_points)
    points = all_points[hull.vertices]

    candidate_normals = get_candidate_normals(points)
    best_volume = np.inf
    best_params = None

    for u in tqdm(candidate_normals):
        # Compute an orthonormal basis: (v, w, u)
        v, w = compute_orthonormal_basis(u)
        # Construct a rotation matrix R whose columns are v, w, u.
        # Then a point p in world coordinates is transformed into
        # p' = [v·p, w·p, u·p].
        R = np.column_stack((v, w, u))
        rotated = points.dot(R)  # (N,3)
        # In the rotated frame, the third coordinate is along u.
        xy = rotated[:, :2]
        z = rotated[:, 2]
        zmin, zmax = z.min(), z.max()
        height_z = zmax - zmin

        # Compute the 2D minimum-area rectangle for the xy–projection.
        rect_coords, area = min_area_rectangle_2d(xy)
        volume = area * height_z

        if volume < best_volume:
            best_volume = volume
            # Determine the orientation in the xy–plane.
            # (Take the first edge of the rectangle.)
            edge = rect_coords[1] - rect_coords[0]
            if np.linalg.norm(edge) < 1e-12:
                continue
            r1_2d = edge / np.linalg.norm(edge)
            # The perpendicular direction in the plane.
            r2_2d = np.array([-r1_2d[1], r1_2d[0]])
            
            # Compute the 2D center (centroid of the rectangle vertices)
            center_xy = rect_coords.mean(axis=0)
            
            # In the rotated frame, the center’s z–coordinate is the midpoint of zmin and zmax.
            center_rot = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2.0])
            
            # The side lengths can be computed from the rectangle vertices.
            side1 = np.linalg.norm(rect_coords[1] - rect_coords[0])
            side2 = np.linalg.norm(rect_coords[2] - rect_coords[1])
            extents_xy = np.array([side1, side2])
            extents = np.array([side1, side2, height_z])
            
            # The box’s local axes in 3D:
            # The first axis r1 is the 2D r1 rotated back into 3D:
            r1 = r1_2d[0] * v + r1_2d[1] * w
            # The second axis is perpendicular: either compute from r2 or as cross(u, r1)
            r2 = r2_2d[0] * v + r2_2d[1] * w
            # The third axis is u.
            R_box = np.column_stack((r1, r2, u))
            
            # To recover the center in world coordinates, note that
            # center_rot = [x_center, y_center, z_center] in the rotated frame.
            # Thus, the world coordinate is:
            center_world = center_rot[0] * v + center_rot[1] * w + center_rot[2] * u
            
            # Compute the 8 corners of the bounding box.
            he = extents / 2.0  # half extents
            corners = []
            for dx in [-he[0], he[0]]:
                for dy in [-he[1], he[1]]:
                    for dz in [-he[2], he[2]]:
                        corner = center_world + dx * r1 + dy * r2 + dz * u
                        corners.append(corner)
            corners = np.array(corners)
            
            best_params = {
                'volume': volume,
                'center': center_world,
                'rotation': R_box,
                'extents': extents,
                'corners': corners
            }

    aligned_points = all_points @ best_params['rotation']
    optimal_volume = best_params['volume']

    return aligned_points, optimal_volume, dict()


# -----------------------------
# Approximate
# -----------------------------

# Function to compute volume of an oriented bounding box
def compute_obb_volume(points, rotation_matrix):
    rotated_points = points @ rotation_matrix.T
    min_corner = rotated_points.min(axis=0)
    max_corner = rotated_points.max(axis=0)
    dimensions = max_corner - min_corner
    return np.prod(dimensions)

# Objective function for optimization
def objective_function(rot_vector, points):
    rotation_matrix = R.from_rotvec(rot_vector).as_matrix()
    return compute_obb_volume(points, rotation_matrix)

# Hybrid Genetic Algorithm + Nelder-Mead Optimization
def optimize_obb(points, population_size=30, generations=200, nelder_mead_iters=10):
    # Generate initial population of random rotations
    population = np.array([np.random.randn(3) for _ in range(population_size)])
    population[0] = np.zeros(3)  # Start with a zero rotation vector
    best_solution = None
    best_volume = float('inf')

    # count_0_volume = float('inf')
    # gen_count = 0
    for generation in range(generations):
        # Evaluate population
        volumes = np.array([objective_function(ind, points) for ind in population])
        
        # Select best candidates
        sorted_indices = np.argsort(volumes)
        population = population[sorted_indices[:population_size // 2]]
        
        # Store best solution
        if volumes[sorted_indices[0]] < best_volume:
            best_volume = volumes[sorted_indices[0]]
            best_solution = population[0]
            # if count_0_volume < float('inf') and (count_0_volume - best_volume) / count_0_volume < 0.01:
            #     gen_count += 1
            # else:
            #     gen_count = 0
            # count_0_volume = best_volume

            # if gen_count > 5:
            #     print(f"Converged after {generation} generations.")
            #     break
        
        # Crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parents_indices = np.random.choice(len(population), 2, replace=False)
            child = (population[parents_indices[0]] + population[parents_indices[1]]) / 2 + np.random.randn(3) * 0.1
            new_population.append(child)
        
        population = np.array(new_population)
    
    # Refine with Nelder-Mead
    result = scipy.optimize.minimize(
        objective_function, best_solution, args=(points,), method='Nelder-Mead', options={'maxiter': nelder_mead_iters}
    )
    
    return result.x, best_volume

# Main function
def compute_optimal_obb(points, population_size=30, generations=200, nelder_mead_iters=10):
    all_points = points.copy()
    hull = scipy.spatial.ConvexHull(points)
    convex_hull_points = points[hull.vertices]
    
    optimal_rotation_vector, optimal_volume = optimize_obb(convex_hull_points, population_size, generations, nelder_mead_iters)
    optimal_rotation_matrix = R.from_rotvec(optimal_rotation_vector).as_matrix()
    
    rotated_points = convex_hull_points @ optimal_rotation_matrix.T
    min_corner = rotated_points.min(axis=0)
    max_corner = rotated_points.max(axis=0)
    dimensions = max_corner - min_corner
    center = (min_corner + max_corner) / 2
    
    aligned_points = all_points @ optimal_rotation_matrix.T

    return aligned_points, optimal_volume, dict()


# -----------------------------
# Approximate optimized
# -----------------------------

# Function to compute volume of an oriented bounding box
@jit(nopython=True)
def compute_obb_volume_jit(points, rotation_matrix):
    rotated_points = np.dot(points, rotation_matrix.T)
    min_corner = np.min(rotated_points, axis=0)
    max_corner = np.max(rotated_points, axis=0)
    dimensions = max_corner - min_corner
    return np.prod(dimensions)

# Objective function for optimization
@jit(nopython=True)
def objective_function_jit(rot_vector, points):
    rotation_matrix = R.from_rotvec(rot_vector).as_matrix()
    return compute_obb_volume_jit(points, rotation_matrix)

# Hybrid Genetic Algorithm + Nelder-Mead Optimization
@jit(nopython=True)
def optimize_obb_jit(points, population_size=30, generations=20, nelder_mead_iters=10):
    # Generate initial population of random rotations
    population = np.array([np.random.randn(3) for _ in range(population_size)])
    
    best_solution = None
    best_volume = float('inf')
    
    for generation in range(generations):
        # Evaluate population
        volumes = np.array([objective_function_jit(ind, points) for ind in population])
        
        # Select best candidates
        sorted_indices = np.argsort(volumes)
        population = population[sorted_indices[:population_size // 2]]
        
        # Store best solution
        if volumes[sorted_indices[0]] < best_volume:
            best_volume = volumes[sorted_indices[0]]
            best_solution = population[0]
        
        # Crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parents_indices = np.random.choice(len(population), 2, replace=False)
            child = (population[parents_indices[0]] + population[parents_indices[1]]) / 2 + np.random.randn(3) * 0.1
            new_population.append(child)
        
        population = np.array(new_population)
    
    # Refine with Nelder-Mead
    result = scipy.optimize.minimize(
        objective_function_jit, best_solution, args=(points,), method='Nelder-Mead', options={'maxiter': nelder_mead_iters}
    )
    
    return result.x, best_volume

# Main function
def compute_optimal_obb_jit(points, population_size=30, generations=20, nelder_mead_iters=10):
    all_points = points.copy()
    hull = scipy.spatial.ConvexHull(points)
    convex_hull_points = points[hull.vertices]
    
    optimal_rotation_vector, optimal_volume = optimize_obb_jit(convex_hull_points, population_size, generations, nelder_mead_iters)
    optimal_rotation_matrix = R.from_rotvec(optimal_rotation_vector).as_matrix()
    
    rotated_points = np.dot(convex_hull_points, optimal_rotation_matrix.T)
    min_corner = np.min(rotated_points, axis=0)
    max_corner = np.max(rotated_points, axis=0)
    dimensions = max_corner - min_corner
    center = (min_corner + max_corner) / 2
    
    aligned_points = np.dot(all_points, optimal_rotation_matrix.T)

    return aligned_points, optimal_volume, dict()

# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    # Create some random 3D points (for example purposes)
    np.random.seed(42)
    pts = np.random.random((1000, 3)).astype(np.float32) * 10. - 5.
    # Optionally, apply a known rotation and translation to test the algorithm:
    angle = np.pi/6
    R_true = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle),  np.cos(angle), 0],
                       [0,              0,             1]])
    pts = pts.dot(R_true.T) + np.array([5, -3, 2])
    
    new_points, _ = minimum_volume_bounding_box(pts)
    new_points_old, _ = basic(pts)