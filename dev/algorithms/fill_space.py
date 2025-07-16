import numpy as np
from scipy.ndimage import binary_closing as binary_closing_scipy
from skimage.morphology import ball, binary_closing

def fill_space(method, solid, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    if method == "none":
        return solid, dict(protein_int_volume=np.sum(solid))
    elif method == "scipy":
        return scipy(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])
    elif method == "skimage_cube":
        return skimage_cube(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])
    elif method == "skimage_ball":
        return skimage_ball(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])
    elif method == "skimage_ball_0.5":
        return skimage_ball(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"] + 0.5 * parameters["resolution"])
    elif method == "propagation_edt":
        return propagation_edt_closing(solid, resolution=parameters["resolution"], probe_radius=parameters["probe_radius"])

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

def skimage_cube(solid, resolution, probe_radius):
    size = np.round(probe_radius / resolution).astype(np.int32)

    structuring_element = ball(size)

    structuring_element[:,:,:] = 1

    expanded_solid = np.zeros((solid.shape[0] + 2*size, solid.shape[1] + 2*size, solid.shape[2] + 2*size), dtype=np.int32)

    expanded_solid[size:-size, size:-size, size:-size] = solid

    solid = binary_closing(expanded_solid, structuring_element, mode="min")[size:-size, size:-size, size:-size]

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)

def skimage_ball(solid, resolution, probe_radius):
    int_radius = np.ceil(probe_radius / resolution).astype(np.int32)

    structuring_element = np.zeros((2 * int_radius + 1, 2 * int_radius + 1, 2 * int_radius + 1), dtype=np.int32)

    # Build the sphere
    for x in range(-int_radius, int_radius + 1):
        for y in range(-int_radius, int_radius + 1):
            for z in range(-int_radius, int_radius + 1):
                if x ** 2 + y ** 2 + z ** 2 <= int_radius ** 2:
                    structuring_element[x + int_radius, y + int_radius, z + int_radius] = 1

    expanded_solid = np.pad(solid, 2*int_radius, mode='constant', constant_values=0)

    print("running binary closing")
    solid = binary_closing(expanded_solid, structuring_element) # , mode="min")
    solid = solid[2*int_radius:-2*int_radius, 2*int_radius:-2*int_radius, 2*int_radius:-2*int_radius]
    print("done")
    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)

connectivity_vectors = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])

def propagation_edt_dilation(solid, distance):
    # Copy the solid
    solid_copy = solid.copy()

    # Find the indices of the solid, where it is equal to 1
    solid_indices = np.argwhere(solid_copy == 1)
    vectors = np.zeros((solid_indices.shape[0], 3))

    c = 0

    while True:
        all_solid_indices = np.repeat(solid_indices, connectivity_vectors.shape[0], axis=0)
        all_vectors = np.repeat(vectors, connectivity_vectors.shape[0], axis=0)
        all_connectivity_vectors = np.repeat(connectivity_vectors.reshape(-1, 1, 3), solid_indices.shape[0], axis=1).transpose(1, 0, 2).reshape(-1, 3)
        adjacent_indices = all_solid_indices + all_connectivity_vectors
        adjacent_vectors = all_vectors + all_connectivity_vectors
        adjacent_norms = np.sum(adjacent_vectors ** 2, axis=1)

        sorting_mask = np.argsort(adjacent_norms)

        sorted_filtered_adjacent_indices = adjacent_indices[sorting_mask][adjacent_norms[sorting_mask] <= distance]
        sorted_filtered_adjacent_vectors = adjacent_vectors[sorting_mask][adjacent_norms[sorting_mask] <= distance]
        
        # Find the indices of the solid, that are adjacent to a voxel equal to 0
        solid_indices, unique_indices = np.unique(sorted_filtered_adjacent_indices[solid_copy[sorted_filtered_adjacent_indices[:, 0], sorted_filtered_adjacent_indices[:, 1], sorted_filtered_adjacent_indices[:, 2]] == 0], axis=0, return_index=True)
        vectors = sorted_filtered_adjacent_vectors[solid_copy[sorted_filtered_adjacent_indices[:, 0], sorted_filtered_adjacent_indices[:, 1], sorted_filtered_adjacent_indices[:, 2]] == 0][unique_indices]
        
        if solid_indices.shape[0] == 0:
            break

        solid_copy[solid_indices[:, 0], solid_indices[:, 1], solid_indices[:, 2]] = 1

    return solid_copy

def propagation_edt_closing(solid, resolution, probe_radius):
    scaled_squared_distance = (probe_radius / resolution) ** 2

    dilated_solid = propagation_edt_dilation(solid, scaled_squared_distance)
    closed_solid = propagation_edt_dilation(1 - dilated_solid, scaled_squared_distance)

    solid = 1 - closed_solid

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)