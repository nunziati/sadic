import numpy as np
from scipy.ndimage.measurements import label

def remove_holes(method, solid, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    last_method = basic

    if method == "original":
        return original(solid)
    elif method == "basic":
        return basic(solid)
    else:
        return last_method(solid)

def original(solid):
    solid.remove_holes()

    return solid, dict()

def basic(solid):
    # import open3d as o3d

    original_voxels = np.sum(solid)
    
    connected_components, n_components = label(1 - solid)

    # for i in range(1, n_components + 1):
    #     count = np.sum(connected_components == i)
    #     print(f"Component {i}: {count} voxels")

    solid = (connected_components != 1).astype(np.int32)
    
    # holes = connected_components * (np.logical_and(connected_components != 0, connected_components != 1))

    # voxel_centers = np.argwhere(holes)
    # values = holes[voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]]

    # # Normalize the values to be between 0 and 1
    # normalized_values = (values - values.min()) / (values.max() - values.min())

    # # Map normalized values to colors (from blue to red)
    # colors = np.zeros((voxel_centers.shape[0], 3))
    # colors[:, 0] = normalized_values  # Red channel
    # colors[:, 2] = 1 - normalized_values  # Blue channel

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    # voxel_centers = np.argwhere(solid)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    # o3d.visualization.draw_geometries([pcd])

    final_voxels = np.sum(solid)

    filled_voxels = final_voxels - original_voxels

    # print(f"Connected components: {n_components}")
    # print(f"Original voxels: {original_voxels}")
    # print(f"Final voxels: {final_voxels}")
    # print(f"Filled voxels: {filled_voxels}")

    return solid, dict(n_components=n_components, n_filled_voxels=filled_voxels, protein_int_volume=final_voxels)