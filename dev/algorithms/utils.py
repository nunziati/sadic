import numpy as np

def cartesian_to_grid(coordinates, extreme_coordinates, resolution):
    return np.floor((coordinates - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

def grid_to_cartesian(coordinates, extreme_coordinates, resolution):
    return coordinates * resolution + extreme_coordinates[:, 0] + resolution / 2

def get_all_coordinate_indexes(grid):
    dimensions = np.array([grid.shape[0], grid.shape[1], grid.shape[2]], dtype=np.int32)
    return (
        np.mgrid[0 : dimensions[0], 0 : dimensions[1], 0 : dimensions[2]]
        .astype(np.int32)
        .transpose(1, 2, 3, 0)
        .reshape(-1, dimensions.shape[0])
    )

def get_all_coordinate_indexes_from_extremes(min_coordinates, max_coordinates):
    return (
        np.mgrid[min_coordinates[0] : max_coordinates[0], min_coordinates[1] : max_coordinates[1], min_coordinates[2] : max_coordinates[2]]
        .astype(np.int32)
        .transpose(1, 2, 3, 0)
        .reshape(-1, min_coordinates.shape[0])
    )