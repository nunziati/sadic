import numpy as np
from scipy.spatial.distance import cdist

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
        
    return depth_idx, dict(p_list = p_list)

def translated_sphere_vectorized(solid, centers, reference_radius, extreme_coordinates, resolution):
    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    p_list = []

    sq_reference_radius = reference_radius ** 2

    for idx, center in enumerate(centers):
        min_coordinates_cartesian = center - reference_radius - 2 * resolution # NOTA BENE QUESTO 2*resolution: l'hai aggiunto ora, è da qui che parte la trasformazione del metodo
        max_coordinates_cartesian = center + reference_radius + 2 * resolution
    
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
        
    return depth_idx, dict(p_list = p_list)