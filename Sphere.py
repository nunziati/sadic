from Solid import Solid
import numpy as np

from numpy.typing import NDArray

class Sphere(Solid):
    def __init__(self, center: NDArray[np.float32], radius: float):
        if center.shape != (3,):
            raise ValueError("must be numpy.ndarray objects with shape (3,)")
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        self.center: NDArray[np.float32] = center
        self.radius: float = radius

    def is_inside(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        return ((points - self.center) ** 2).sum(axis=1) <= self.radius ** 2
    
    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        return np.stack(((self.center - self.radius), (self.center + self.radius)), axis=-1)