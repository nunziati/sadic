import numpy as np
from scipy.ndimage import binary_closing
from skimage.morphology import ball, closing, reconstruction, erosion

def fill_space(method, solid, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    last_method = scipy

    if method == "none":
        return solid, dict()
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
    solid = binary_closing(solid, structure=structuring_element)

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)

def skimage(solid, resolution, probe_radius):
    # Create the spherical structuring element
    structuring_element = ball(int(probe_radius / resolution))

    # Fill the space
    solid = closing(solid, structuring_element)

    protein_int_volume = np.sum(solid)

    return solid, dict(protein_int_volume=protein_int_volume)