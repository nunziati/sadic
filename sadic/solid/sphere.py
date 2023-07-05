r"""Defines the Sphere class."""

import numpy as np
from numpy.typing import NDArray

from sadic.solid import Solid


class Sphere(Solid):
    r"""Solid representing a sphere.

    Can be used to check if a (set of) point(s) is inside the sphere. Can also be used to get the
    extreme coordinates of the sphere.

    Attributes:
        center (NDArray[np.float32]):
            A numpy array of floats representing the center of the sphere.
        radius (float):
            A float representing the radius of the sphere.

    Methods:
        __init__:
            Constructor for spheres.
        is_inside:
            Method to check if a (set of) point(s) is inside the sphere.
        get_extreme_coordinates:
            Method to get the extreme coordinates of the sphere.
    """

    def __init__(self, center: NDArray[np.float32], radius: float) -> None:
        r"""Initializes a sphere with the given center and radius.

        Args:
            center (NDArray[np.float32]):
                A numpy array of floats representing the center of the sphere.
            radius (float):
                A float representing the radius of the sphere.

        Raises:
            ValueError:
                If the center is not a numpy array of floats with shape (3,) or if the radius is not
                a positive float.
        """
        if center.shape != (3,):
            raise ValueError(
                "must be numpy.ndarray objects with shape (3,) " f"(got {center.shape})"
            )
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        self.center: NDArray[np.float32] = center
        self.radius: float = radius

    def is_inside(self, *args, **kwargs) -> NDArray[np.bool_]:
        r"""Checks if a (set of) point(s) is inside the sphere.

        Args:
            points (NDArray[np.float32]):
                A numpy array of floats representing the point(s) to check.
            radii (NDArray[np.float32] | None):
                A numpy array of floats representing the radii of the point(s) to check. If None,
                the radii are assumed to be zero.

        Returns (NDArray[np.bool_]):
            A numpy array of booleans representing whether the point(s) is inside the sphere.
        """
        points: NDArray[np.float32] = args[0]
        radii: NDArray[np.float32] | None = kwargs.get("radii", None)
        if radii is None:
            return ((points - self.center) ** 2).sum(axis=1) <= self.radius**2

        return ((points - self.center) ** 2).sum(axis=1) <= (self.radius - radii) ** 2

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        r"""Gets the extreme coordinates of the sphere.

        Returns (NDArray[np.float32]):
            A numpy array of floats representing the extreme coordinates of the sphere.
        """
        return np.stack(((self.center - self.radius), (self.center + self.radius)), axis=-1)
