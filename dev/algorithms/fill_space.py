import numpy as np
from scipy.ndimage import binary_closing as binary_closing_scipy
from skimage.morphology import ball, binary_closing, reconstruction, erosion

def fill_space(method, solid, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    last_method = scipy

    if method == "none":
        return solid, dict(protein_int_volume=np.sum(solid))
    elif method == "scipy":
        return scipy(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])
    elif method == "skimage":
        return skimage(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])
    else:
        return last_method(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])

def create_spherical_structuring_element(radius):
    # Calculate the size of the grid
    L = int(2 * radius + 1)
    structuring_element = np.zeros((L, L, L), dtype=bool)

    # Calculate the center of the sphere
    center = radius

    # Iterate over the grid and fill in the sphere
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) <= radius:
                    structuring_element[x, y, z] = 1

    return structuring_element

def scipy(solid, resolution, probe_radius):
    # Create the spherical structuring element
    structuring_element = create_spherical_structuring_element(int(probe_radius / resolution))

    # Fill the space
    solid = binary_closing_scipy(solid, structure=structuring_element)

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)

def skimage(solid, resolution, probe_radius):
    # from skimage.morphology import disk
    # import open3d as o3d

    # Create the spherical structuring element
    structuring_element = ball(np.ceil(probe_radius / resolution))

    # Plot the solid vefore closing, using open3d
    # voxel_centers = np.argwhere(solid)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    # # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    # Fill the space
    solid = binary_closing(solid, structuring_element)

    # # Plot the solid after closing, using open3d
    # voxel_centers = np.argwhere(solid)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    # # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)