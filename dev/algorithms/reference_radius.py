import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from sadic.algorithm.radius import find_max_radius_point_voxel
from .utils import cartesian_to_grid, grid_to_cartesian

def find_reference_radius(method, solid, atoms, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    if method == "original":
        return original(solid, atoms)
    elif method == "basic":
        return basic(solid, atoms, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "basic_vectorized":
        return basic_vectorized(solid, atoms, parameters["extreme_coordinates"], parameters["resolution"])

def original(solid, atoms):
    reference_radius = find_max_radius_point_voxel(solid)[1]

    return reference_radius, dict()

def basic(solid, atoms, extreme_coordinates, resolution):
    centers_indexes = np.empty_like(atoms, dtype=np.int32)
    for idx, atom in enumerate(atoms):
        centers_indexes[idx] = cartesian_to_grid(atom, extreme_coordinates, resolution)

    edt = distance_transform_edt(solid, sampling=resolution)

    edt_centers = edt[
        centers_indexes[:, 0], centers_indexes[:, 1], centers_indexes[:, 2]
    ]

    max_edt_center = int(np.argmax(edt_centers))
    max_edt = edt_centers[max_edt_center]

    return max_edt, dict()

def basic_vectorized(solid, atoms, extreme_coordinates, resolution):
    centers_indexes = cartesian_to_grid(atoms, extreme_coordinates, resolution)

    edt = distance_transform_edt(solid, sampling=resolution)

    edt_centers = edt[
        centers_indexes[:, 0], centers_indexes[:, 1], centers_indexes[:, 2]
    ]

    max_edt_center = int(np.argmax(edt_centers))
    max_edt = edt_centers[max_edt_center]

    return max_edt, dict()