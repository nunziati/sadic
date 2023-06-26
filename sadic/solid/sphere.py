import numpy as np
from numpy.typing import NDArray

from sadic.solid import Solid

class Sphere(Solid):
    def __init__(self, center: NDArray[np.float32], radius: float):
        if center.shape != (3,):
            raise ValueError(
                "must be numpy.ndarray objects with shape (3,) "
                f"(got {center.shape})")
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        self.center: NDArray[np.float32] = center
        self.radius: float = radius

    def is_inside(
            self,
            points: NDArray[np.float32],
            radii: NDArray[np.float32] | None = None) -> NDArray[np.bool_]:
        if radii is None:
            return ((points - self.center) ** 2
                    ).sum(axis=1) <= self.radius ** 2
        else:    
            return ((points - self.center) ** 2
                ).sum(axis=1) <= (self.radius - radii) ** 2
    
    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        return np.stack(((self.center - self.radius),
                         (self.center + self.radius)), axis=-1)