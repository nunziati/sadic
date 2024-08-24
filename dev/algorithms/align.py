import numpy as np
import open3d as o3d

def align(method, points, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    last_method = basic

    if method == "none":
        return points, dict()
    elif method == "basic":
        return basic(points)
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

    return aligned_points, dict()