import numpy as np
from scipy.ndimage.measurements import label

def remove_holes(method, solid, **parameters):
    """
    Returns:
    - solid: np.ndarray
    - complexity_variables: dict
    """

    if method == "original":
        return original(solid)
    elif method == "basic":
        return basic(solid)
    elif method == "basic_vectorized":
        return basic(solid)
    elif method == "translated_sphere_vectorized":
        return basic(solid)

def original(solid):
    solid.remove_holes()

    return solid, dict()

def basic(solid):
    original_voxels = np.sum(solid)
    connected_components, n_components = label(solid)
    solid = (connected_components != 0).astype(np.int32)
    final_voxels = np.sum(solid)

    filled_voxels = final_voxels - original_voxels

    return solid, dict(n_components=n_components, n_filled_voxels=filled_voxels, n_protein_int_volume=final_voxels)