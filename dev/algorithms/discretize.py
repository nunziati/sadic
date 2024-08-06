import numpy as np
from scipy.spatial.distance import cdist

from .utils import grid_to_cartesian, get_all_coordinate_indexes_from_extremes
from sadic.solid import VoxelSolid

def discretize(method, model, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    if method == "original":
        return original(model, parameters["resolution"])
    elif method == "basic":
        return basic(model, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "basic_vectorized":
        return basic_vectorized(model, parameters["extreme_coordinates"], parameters["resolution"])

def original(model, resolution):
    solid = VoxelSolid(model, resolution=resolution)

    return solid, dict()

def basic(model, extreme_coordinates, resolution):
    centers = model["atoms"]
    radii = model["radii"]

    dimensions = np.ceil(
        (extreme_coordinates[:, 1] - extreme_coordinates[:, 0]) / resolution
    ).astype(np.int32)
    
    solid = np.full(dimensions, 0, dtype=np.int32)

    n = 0

    # this for loop is N*R_wdw_max^3
    for center, radius in zip(centers, radii):
        sqradius = radius ** 2
        min_coordinates = np.floor((center - radius - extreme_coordinates[:, 0]) / resolution).astype(np.int32)
        max_coordinates = np.ceil((center + radius - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

        min_overlapping_coordinates = np.maximum(min_coordinates, np.array([0, 0, 0]))
        max_overlapping_coordinates = np.minimum(max_coordinates, dimensions)
        for x in range(min_overlapping_coordinates[0], max_overlapping_coordinates[0]):
            for y in range(min_overlapping_coordinates[1], max_overlapping_coordinates[1]):
                for z in range(min_overlapping_coordinates[2], max_overlapping_coordinates[2]):
                    n += 1
                    if (
                        cdist(
                            center.reshape(-1, 3),
                            grid_to_cartesian(np.array([x, y, z]), extreme_coordinates, resolution).reshape(-1, 3),
                            metric="sqeuclidean")[0]
                        <= sqradius
                    ):
                        solid[x, y, z] = 1

    return solid, dict(n=n)

def basic_vectorized(model, extreme_coordinates, resolution):
    centers = model["atoms"]
    radii = model["radii"]

    dimensions = np.ceil(
        (extreme_coordinates[:, 1] - extreme_coordinates[:, 0]) / resolution
    ).astype(np.int32)
    
    solid = np.full(dimensions, 0, dtype=np.int32)

    n = 0

    # this for loop is N*R_wdw_max^3
    for center, radius in zip(centers, radii):
        sqradius = radius ** 2
        min_coordinates = np.floor((center - radius - extreme_coordinates[:, 0]) / resolution).astype(np.int32)
        max_coordinates = np.ceil((center + radius - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

        min_overlapping_coordinates = np.maximum(min_coordinates, np.array([0, 0, 0]))
        max_overlapping_coordinates = np.minimum(max_coordinates, dimensions)

        sphere_view = solid[min_overlapping_coordinates[0]:max_overlapping_coordinates[0], min_overlapping_coordinates[1]:max_overlapping_coordinates[1], min_overlapping_coordinates[2]:max_overlapping_coordinates[2]]
        
        overlapping_coordinates = get_all_coordinate_indexes_from_extremes(min_overlapping_coordinates, max_overlapping_coordinates)
        overlapping_cartesian_coordinates = grid_to_cartesian(overlapping_coordinates, extreme_coordinates, resolution)
        scaled_overlapping_coordinates = overlapping_coordinates - min_overlapping_coordinates
        sphere_overlap = cdist(
            center.reshape(-1, 3),
            overlapping_cartesian_coordinates,
            metric="sqeuclidean"
        )[0] <= sqradius

        solid[overlapping_coordinates[:, 0], overlapping_coordinates[:, 1], overlapping_coordinates[:, 2]] = (
            np.logical_or(sphere_view[scaled_overlapping_coordinates[:, 0], scaled_overlapping_coordinates[:, 1], scaled_overlapping_coordinates[:, 2]], sphere_overlap).astype(np.int32).reshape(-1)
        )

    return solid, dict(n=n)