import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import open3d as o3d
from sklearn.neighbors import KDTree as KDTree_sklearn
from sklearn.neighbors import BallTree

from sadic.algorithm.depth import sadic_original_voxel
from .utils import cartesian_to_grid, grid_to_cartesian, get_all_coordinate_indexes, get_all_coordinate_indexes_from_extremes

def compute_indexes(method, solid, centers, reference_radius, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    if method == "original":
        return original(solid, centers, reference_radius, parameters["model"])
    elif method == "basic":
        return basic(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "basic_vectorized":
        return basic_vectorized(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "translated_sphere_vectorized":
        return translated_sphere_vectorized(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "coeurjolly_translated_sphere":
        return translated_sphere_vectorized(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "octrees":
        return oc_trees(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "kdtrees_scipy":
        return kd_trees_scipy(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "kdtrees_open3d":
        return kd_trees_open3d(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "kdtrees_sklearn":
        return kd_trees_sklearn(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "balltrees_sklearn":
        return ball_trees_sklearn(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "octrees_range_query":
        return oc_trees_range_query(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "kdtrees_voxel_scan_loop":
        return kd_trees_voxel_scan_loop(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "kd_trees_scipy_tree_comparison":
        return kd_trees_scipy_tree_comparison(solid, centers, reference_radius, parameters["extreme_coordinates"], parameters["resolution"])

    

def original(solid, centers, reference_radius, model):
    result = sadic_original_voxel(solid, model, reference_radius)

    return result, dict()

def basic(solid, centers, reference_radius, extreme_coordinates, resolution):
    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    p_list = []

    sq_reference_radius = reference_radius ** 2

    for idx, center in enumerate(centers):
        min_coordinates_cartesian = center - reference_radius
        max_coordinates_cartesian = center + reference_radius
        min_coordinates = np.floor((min_coordinates_cartesian - extreme_coordinates[:, 0]) / resolution).astype(np.int32)
        max_coordinates = np.ceil((max_coordinates_cartesian - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

        sphere_box = np.zeros((max_coordinates[0] - min_coordinates[0], max_coordinates[1] - min_coordinates[1], max_coordinates[2] - min_coordinates[2]), dtype=np.int32)

        sphere_int_volume = 0
        intersection_int_volume = 0

        for i in range(min_coordinates[0], max_coordinates[0]):
            for j in range(min_coordinates[1], max_coordinates[1]):
                for k in range(min_coordinates[2], max_coordinates[2]):
                    ijk_solid = np.array([i, j, k])
                    xyz_solid = grid_to_cartesian(ijk_solid, extreme_coordinates, resolution)
                    if cdist(xyz_solid.reshape(-1, 3), center.reshape(-1, 3), metric="sqeuclidean")[0] <= sq_reference_radius:
                        sphere_int_volume += 1
                        # check if i,j,k is inside the solid grid and if it is inside the solid
                        if (
                            (i >= 0)
                            and (i < solid.shape[0])
                            and (j >= 0)
                            and (j < solid.shape[1])
                            and (k >= 0)
                            and (k < solid.shape[2])
                            and solid[i, j, k]
                        ):
                            intersection_int_volume += 1

        depth_idx[idx] = 2 * (1 - intersection_int_volume / sphere_int_volume)

        p_list.append(sphere_box.shape[0] * sphere_box.shape[1] * sphere_box.shape[2])
        
    return depth_idx, dict(p_list = p_list)

def basic_vectorized(solid, centers, reference_radius, extreme_coordinates, resolution):
    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    p_list = []
    voxel_operations_map = np.zeros_like(solid, dtype=np.int32)

    sq_reference_radius = reference_radius ** 2

    for idx, center in enumerate(centers):
        min_coordinates_cartesian = center - reference_radius
        max_coordinates_cartesian = center + reference_radius
        min_coordinates = np.floor((min_coordinates_cartesian - extreme_coordinates[:, 0]) / resolution).astype(np.int32)
        max_coordinates = np.ceil((max_coordinates_cartesian - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

        sphere_box_dimensions = max_coordinates - min_coordinates
        sphere_box = np.zeros(sphere_box_dimensions, dtype=np.int32)

        sphere_int_volume = 0
        intersection_int_volume = 0

        sphere_grid_coordinates = get_all_coordinate_indexes_from_extremes(min_coordinates, max_coordinates)
        sphere_cartesian_coordinates = grid_to_cartesian(sphere_grid_coordinates, extreme_coordinates, resolution)
        sphere_box = np.where(cdist(sphere_cartesian_coordinates, center.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape(sphere_box_dimensions)

        sphere_int_volume = np.count_nonzero(sphere_box)

        solid_min_overlapping_coordinates = np.maximum(min_coordinates, np.array([0, 0, 0]))
        solid_max_overlapping_coordinates = np.minimum(max_coordinates, np.array([solid.shape[0], solid.shape[1], solid.shape[2]]))

        sphere_min_overlapping_coordinates = np.maximum(-min_coordinates, np.array([0, 0, 0]))
        sphere_max_overlapping_coordinates = np.minimum(solid.shape - min_coordinates, sphere_box_dimensions)

        solid_overlap = solid[solid_min_overlapping_coordinates[0]:solid_max_overlapping_coordinates[0], solid_min_overlapping_coordinates[1]:solid_max_overlapping_coordinates[1], solid_min_overlapping_coordinates[2]:solid_max_overlapping_coordinates[2]]
        sphere_overlap = sphere_box[sphere_min_overlapping_coordinates[0]:sphere_max_overlapping_coordinates[0], sphere_min_overlapping_coordinates[1]:sphere_max_overlapping_coordinates[1], sphere_min_overlapping_coordinates[2]:sphere_max_overlapping_coordinates[2]]

        intersection_int_volume = np.count_nonzero(np.logical_and(solid_overlap, sphere_overlap))

        depth_idx[idx] = 2 * (1 - intersection_int_volume / sphere_int_volume)

        p_list.append(sphere_box.shape[0] * sphere_box.shape[1] * sphere_box.shape[2])
        voxel_operations_map[solid_min_overlapping_coordinates[0]:solid_max_overlapping_coordinates[0], solid_min_overlapping_coordinates[1]:solid_max_overlapping_coordinates[1], solid_min_overlapping_coordinates[2]:solid_max_overlapping_coordinates[2]] += 1

    return depth_idx, dict(p_list = p_list, voxel_operations_map = voxel_operations_map)

def translated_sphere_vectorized(solid, centers, reference_radius, extreme_coordinates, resolution):
    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    p_list = []
    voxel_operations_map = np.zeros_like(solid, dtype=np.int32)

    sq_reference_radius = reference_radius ** 2
    
    solid_dimensions = np.array([solid.shape[0], solid.shape[1], solid.shape[2]], dtype=np.int32)
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)

    for idx, center in enumerate(centers):
        grid_center = cartesian_to_grid(center, extreme_coordinates, resolution)

        solid_min_overlapping_coordinates = np.maximum(grid_center - sphere_grid_radius, 0)
        solid_max_overlapping_coordinates = np.minimum(grid_center + sphere_grid_radius + 1, solid_dimensions)
        
        sphere_min_overlapping_coordinates = np.maximum(grid_origin - grid_center, 0)
        sphere_max_overlapping_coordinates = np.minimum(sphere_grid_radius + (solid_dimensions - grid_center), sphere_grid_diameter)
        
        solid_overlap = solid[solid_min_overlapping_coordinates[0]:solid_max_overlapping_coordinates[0], solid_min_overlapping_coordinates[1]:solid_max_overlapping_coordinates[1], solid_min_overlapping_coordinates[2]:solid_max_overlapping_coordinates[2]]
        sphere_overlap = sphere_box[sphere_min_overlapping_coordinates[0]:sphere_max_overlapping_coordinates[0], sphere_min_overlapping_coordinates[1]:sphere_max_overlapping_coordinates[1], sphere_min_overlapping_coordinates[2]:sphere_max_overlapping_coordinates[2]]

        intersection_int_volume = np.count_nonzero(np.logical_and(solid_overlap, sphere_overlap))

        depth_idx[idx] = 2 * (1 - intersection_int_volume / sphere_int_volume)
        
        p_list.append(solid_overlap.shape[0] * solid_overlap.shape[1] * solid_overlap.shape[2])
        voxel_operations_map[solid_min_overlapping_coordinates[0]:solid_max_overlapping_coordinates[0], solid_min_overlapping_coordinates[1]:solid_max_overlapping_coordinates[1], solid_min_overlapping_coordinates[2]:solid_max_overlapping_coordinates[2]] += 1

    return depth_idx, dict(p_list = p_list, voxel_operations_map = voxel_operations_map)


def oc_trees(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("function starts at least", flush=True)
    def voxel_grid_to_octree(voxel_grid, max_depth=10):
        coords = np.array(np.nonzero(voxel_grid)).T
        points = coords.astype(np.float64)
        pc =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        octree = o3d.geometry.Octree(max_depth=max_depth)
        octree.convert_from_point_cloud(pc, size_expand=0.01)
        
        return octree
    
    def compute_octree_intersection_volume(octree1, octree2):
        # Compute the intersection volume of two octrees
        intersection_volume = 0.0
        for node1 in octree1.get_leaf_nodes():
            for node2 in octree2.get_leaf_nodes():
                bbox1 = node1.get_bounding_box()
                bbox2 = node2.get_bounding_box()
                
                # Compute intersection of bounding boxes
                inter_bbox = bbox1.intersection(bbox2)
                if inter_bbox is not None:
                    intersection_volume += inter_bbox.volume()
        
        return intersection_volume
    
    solid_octree = voxel_grid_to_octree(solid)

    # in a voxel grid of the same dimension as the solid, create a (full) sphere with the same radius as the reference radius, centered at the center and compute its octree
    sphere_voxel_grid = np.zeros_like(solid)
    build_center = np.array([solid.shape[0] // 2, solid.shape[1] // 2, solid.shape[2] // 2])
    sphere_radius = reference_radius / resolution
    sphere_grid_coordinates = get_all_coordinate_indexes(sphere_voxel_grid)
    sphere_voxel_grid = np.where(cdist(sphere_grid_coordinates, build_center.reshape(-1, 3), metric="sqeuclidean") <= sphere_radius ** 2, 1, 0).reshape(sphere_voxel_grid.shape)

    sphere_volume = np.count_nonzero(sphere_voxel_grid)

    sphere_octree = voxel_grid_to_octree(sphere_voxel_grid)

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    print("Sphere volume:", sphere_volume)

    for idx, center in enumerate(centers):
        # translate the sphere octree to the center
        sphere_octree.translate(center, False)

        intersection_volume = compute_octree_intersection_volume(solid_octree, sphere_octree)

        print("Intersection volume:", intersection_volume)

        depth_idx[idx] = 2 * (1 - intersection_volume / sphere_volume)

        print("Depth index:", depth_idx[idx])

    return depth_idx, dict()

def kd_trees_scipy(solid, centers, reference_radius, extreme_coordinates, resolution):
    def count_points_within_radius(kdtree, point, radius):
        # Find indices of points within the radius
        indices = kdtree.query_ball_point(point, radius)
        
        # The count is the number of points found within the radius
        count = len(indices)
        return count

    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    solid_points = np.array(np.nonzero(solid)).T
    kdtree = KDTree(solid_points, leafsize=200)
    print("wow3", flush=True)
    
    count = np.empty(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    for idx, grid_center in enumerate(grid_centers):
        count[idx] = count_points_within_radius(kdtree, grid_center, sphere_grid_radius)

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()


def kd_trees_open3d(solid, centers, reference_radius, extreme_coordinates, resolution):
    def count_points_within_radius(kdtree, point, radius):
        k, _, _ = kdtree.search_radius_vector_3d(point, radius)
        return k

    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)

    solid_points = np.array(np.nonzero(solid)).T
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(solid_points.astype(np.float64)))
    kdtree = o3d.geometry.KDTreeFlann(pc)

    count = np.empty(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    for idx, grid_center in enumerate(grid_centers):
        count[idx] = count_points_within_radius(kdtree, grid_center, sphere_grid_radius)

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()


def kd_trees_sklearn(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    solid_points = np.array(np.nonzero(solid)).T
    kdtree = KDTree_sklearn(solid_points, leaf_size=200)
    print("wow3", flush=True)
    
    count = np.empty(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    count = kdtree.query_radius(grid_centers, r=sphere_grid_radius, count_only=True)

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()

def kd_trees_scipy_tree_comparison(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    count = np.empty(centers.shape[0], dtype=np.int32)

    solid_points = np.array(np.nonzero(solid)).T
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)
    kdtree = KDTree(solid_points, leafsize=200)
    query_kdtree = KDTree(grid_centers, leafsize=40)
    print("wow3", flush=True)
    
    results = query_kdtree.query_ball_tree(kdtree, sphere_grid_radius)

    for idx, result in enumerate(results):
        count[idx] = len(result)

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()

def kd_trees_voxel_scan_loop(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    solid_points = np.array(np.nonzero(solid)).T
    
    count = np.zeros(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    kdtree = KDTree_sklearn(grid_centers, leaf_size=1)
    print("wow3", flush=True)

    from tqdm import tqdm

    for idx, point in tqdm(enumerate(solid_points)):
        near_centers = kdtree.query_radius([point], r=sphere_grid_radius, return_distance=False)[0]
        count[near_centers] += 1

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()

def kd_trees_scipy_voxel_scan_loop(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    solid_points = np.array(np.nonzero(solid)).T
    
    count = np.zeros(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    kdtree = KDTree(grid_centers, leafsize=1)
    print("wow3", flush=True)

    from tqdm import tqdm

    for idx, point in tqdm(enumerate(solid_points)):
        near_centers = kdtree.query_radius([point], r=sphere_grid_radius, return_distance=False)[0]
        count[near_centers] += 1

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()

def ball_trees_sklearn(solid, centers, reference_radius, extreme_coordinates, resolution):
    print("wow1", flush=True)
    sq_reference_radius = reference_radius ** 2
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)
    print("wow2", flush=True)

    solid_points = np.array(np.nonzero(solid)).T
    kdtree = BallTree(solid_points, leaf_size=400)
    print("wow3", flush=True)
    
    count = np.empty(centers.shape[0], dtype=np.int32)
    
    grid_centers = cartesian_to_grid(centers, extreme_coordinates, resolution)

    count = kdtree.query_radius(grid_centers, r=sphere_grid_radius, count_only=True)

    depth_idx = 2 * (1 - count / sphere_int_volume)

    return depth_idx, dict()

def oc_trees_range_query(solid, centers, reference_radius, extreme_coordinates, resolution):
    def voxel_grid_to_octree(voxel_grid, max_depth=10):
        print("0", flush=True)
        coords = np.array(np.nonzero(voxel_grid)).T
        print("1", flush=True)
        points = coords.astype(np.float64)
        print("2", flush=True)
        pc =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        print("3", flush=True)
        octree = o3d.geometry.Octree(max_depth=max_depth)
        print("4", flush=True)
        octree.convert_from_voxel_grid(pc, size_expand=0.01)
        print("5", flush=True)
        
        return octree
    
    def count_points_within_radius(octree, point, radius):
        # Initialize the count of points
        count = 0
        
        # Define a callback function for traversing the octree
        def traverse_callback(node, node_info):
            nonlocal count
            if node.is_leaf:
                # Get the list of points in this leaf node
                points = np.asarray(node.indices)
                # Get the actual points' coordinates
                points_coords = np.asarray(octree.points)[points]
                # Calculate the distance from the reference point to all points in this leaf
                distances = np.linalg.norm(points_coords - point, axis=1)
                # Count the number of points within the radius
                count += np.sum(distances <= radius)

        # Traverse the octree
        octree.traverse(traverse_callback)
        
        return count
    
    print("octree construction", flush=True)
    solid_octree = voxel_grid_to_octree(solid)
    print("octree construction done", flush=True)

    # in a voxel grid of the same dimension as the solid, create a (full) sphere with the same radius as the reference radius, centered at the center and compute its octree
    sphere_voxel_grid = np.zeros_like(solid)
    build_center = np.array([solid.shape[0] // 2, solid.shape[1] // 2, solid.shape[2] // 2])
    sphere_radius = reference_radius / resolution
    sphere_grid_coordinates = get_all_coordinate_indexes(sphere_voxel_grid)
    sphere_voxel_grid = np.where(cdist(sphere_grid_coordinates, build_center.reshape(-1, 3), metric="sqeuclidean") <= sphere_radius ** 2, 1, 0).reshape(sphere_voxel_grid.shape)

    sphere_volume = np.count_nonzero(sphere_voxel_grid)

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, center in enumerate(centers):
        intersection_volume = count_points_within_radius(solid_octree, center, sphere_radius)

        depth_idx[idx] = 2 * (1 - intersection_volume / sphere_volume)

    return depth_idx, dict()