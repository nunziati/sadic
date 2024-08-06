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

def original(solid):
    solid.remove_holes()

    return solid, dict()

def basic(solid):
    solid = (label(solid)[0] != 0).astype(np.int32)

    return solid, dict()