import sys

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from sadic.algorithm.radius import find_max_radius_point_voxel
from .utils import cartesian_to_grid, grid_to_cartesian

# sys.path.append("/home/giacomo/sadic/dev/cpp_tests/DGtalBind/build")
# import my_module

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
    elif method == "translated_sphere_vectorized":
        return basic_vectorized(solid, atoms, parameters["extreme_coordinates"], parameters["resolution"])
    elif method == "translated_sphere_vectorized_0.5":
        radius, diz = basic_vectorized(solid, atoms, parameters["extreme_coordinates"], parameters["resolution"])
        return radius + 0.5 * parameters["resolution"], diz
    # elif method == "coeurjolly_translated_sphere":
    #     return coeurjolly_translated_sphere(solid, atoms, parameters["extreme_coordinates"], parameters["resolution"])
    
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

    if not (centers_indexes >= 0).all():
        raise ValueError("Some atoms are outside the grid defined by extreme_coordinates and resolution.")
    
    edt = distance_transform_edt(solid, sampling=resolution)

    edt_centers = edt[
        centers_indexes[:, 0], centers_indexes[:, 1], centers_indexes[:, 2]
    ]

    max_edt_center = int(np.argmax(edt_centers))
    max_edt = edt_centers[max_edt_center]

    return max_edt, dict()

# def coeurjolly_translated_sphere(solid, atoms, extreme_coordinates, resolution):
#     centers_indexes = cartesian_to_grid(atoms, extreme_coordinates, resolution)

#     max_edt = my_module.compute_max_distance(solid, centers_indexes)

#     max_edt = max_edt * resolution

#     return max_edt, dict()