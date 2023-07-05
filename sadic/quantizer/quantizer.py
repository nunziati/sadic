r"""Defines the Quantizer abstract class and its specialized subclasses."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sadic.solid import Solid, Sphere
from sadic.utils import Repr


class Quantizer(ABC, Repr):
    r"""Abstract class for quantizers.

    A quantizer is an object that quantizes a solid into a set of points and volumes. Possibly it
    can also perform other quantization-related tasks, such as surface quantization.

    Attributes:
        None

    Methods:
        __init__:
            Abstract method that initializes the Quantizer object.
        get_points_and_volumes:
            Abstract method that returns the representative points and volumes of the quantized
            solid.
        get_extreme_coordinates:
            Returns the extreme coordinates of a solid, if available.
    """

    @abstractmethod
    def __init__(self) -> None:
        r"""Abstract method that initializes the Quantizer object."""

    @abstractmethod
    def get_points_and_volumes(
        self, solid: Solid
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Abstract method, returns the points and volumes of a solid.

        Args:
            solid (Solid):
                A Solid object that will be quantized.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            The first array contains the points of the quantized solid, and the second array
            contains the volumes of the quantized cells.
        """

    def get_extreme_coordinates(self, solid: Solid) -> NDArray[np.float32]:
        r"""Returns the extreme coordinates of a solid, if available.

        This method checks if the Solid object has a 'get_extreme_coordinates' method. If it does,
        it calls that method and returns the result. If not, it raises a NotImplementedError.

        Args:
            solid (Solid):
                A Solid object for which to get the extreme coordinates.

        Returns:
            NDArray[np.float32]: An array containing the extreme coordinates of the solid.

        Raises:
            NotImplementedError: If the Solid object does not have a 'get_extreme_coordinates'
            method.

        """
        if hasattr(solid, "get_extreme_coordinates"):
            return solid.get_extreme_coordinates()

        raise NotImplementedError


class CartesianQuantizer(Quantizer):
    r"""Abstract class for Cartesian quantizers.

    A Cartesian quantizer is an object that quantizes a solid into a set of points and volumes using
    a Cartesian coordinate system, resulting in a Cartesian grid.

    Attributes:
        None

    Methods:
        __init__:
            Abstract method that initializes the CartesianQuantizer object.
        get_points_and_volumes:
            Abstract method that returns the representative points and volumes of the quantized
            solid.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Abstract method that initializes the CartesianQuantizer object."""

    @abstractmethod
    def get_points_and_volumes(
        self, solid: Solid
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Abstract method, returns the representative points and volumes of the quantized solid.

        Args:
            solid (Solid):
                A Solid object that will be quantized.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            The first array contains the points of the quantized solid, and the second array
            contains the volumes of the quantized cells.
        """


class RegularStepsCartesianQuantizer(CartesianQuantizer):
    r"""Quantizer that quantizes a solid on a grid with fixed dimension.

    Attributes:
        steps_number_<i> (int):
            Number of grid cells along dimension i, with i in [x, y, z].

    Methods:
        __init__:
            Initializes the RegularStepsCartesianQuantizer object with the given number of steps
            along each dimension.
        get_points_and_volumes:
            Returns the representative points and volumes of the quantized solid.
    """

    def __init__(self, steps_number: int | tuple[int, int, int]) -> None:
        r"""Initializes the quantizer object with the given number of steps along each dimension.

        Args:
            steps_number (int | tuple[int, int, int]):
                Number of steps along each dimension. If an integer is given, the same number of
                steps is used along each dimension. If a tuple of three integers is given, the
                first integer is used for the x dimension, the second integer is used for the y
                dimension, and the third integer is used for the z dimension.

        Raises:
            ValueError: If the number of steps along any dimension is less than 1.
        """
        x_steps_number: int
        y_steps_number: int
        z_steps_number: int

        if isinstance(steps_number, int):
            x_steps_number = y_steps_number = z_steps_number = steps_number
        else:
            x_steps_number, y_steps_number, z_steps_number = steps_number

        step_type: str
        step_number: int
        for step_type, step_number in zip(
            ("x", "y", "z"), (x_steps_number, y_steps_number, z_steps_number)
        ):
            if step_number < 1:
                raise ValueError(f"{step_type} step number must be greater than 0")

        self.x_steps_number: int = x_steps_number
        self.y_steps_number: int = y_steps_number
        self.z_steps_number: int = z_steps_number

    def get_points_and_volumes(
        self, solid: Solid
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Returns the representative points and volumes of the quantized solid.

        Args:
            solid (Solid):
                A Solid object that will be quantized.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            The first array contains the points of the quantized solid, and the second array
            contains the volumes of the quantized cells.
        """
        extremes: NDArray[np.float32] = self.get_extreme_coordinates(solid)
        x_values: NDArray[np.float32] = np.linspace(
            extremes[0][0], extremes[0][1], self.x_steps_number, dtype=np.float32
        )
        y_values: NDArray[np.float32] = np.linspace(
            extremes[1][0], extremes[1][1], self.y_steps_number, dtype=np.float32
        )
        z_values: NDArray[np.float32] = np.linspace(
            extremes[2][0], extremes[2][1], self.z_steps_number, dtype=np.float32
        )

        volume: NDArray[np.float32] = (
            (x_values[1] - x_values[0]) * (y_values[1] - y_values[0]) * (z_values[1] - z_values[0])
        )

        grid: NDArray[np.float32] = np.stack(
            np.meshgrid(x_values, y_values, z_values), axis=-1
        ).reshape((-1, 3))

        grid_inside: NDArray[np.float32] = grid[solid.is_inside(grid)]

        return (grid_inside, np.full((grid_inside.shape[0],), volume, dtype=np.float32))


class RegularSizeCartesianQuantizer(CartesianQuantizer):
    r"""Quantizer that quantizes a solid on a grid with fixed cell size.

    To be implemented.

    Attributes:
        cell_size_<i> (float):
            Size of grid cells along dimension i, with i in [x, y, z].

    Methods:
        __init__:
            Initializes the RegularSizeCartesianQuantizer object with the given cell size along
            each dimension.
        get_points_and_volumes:
            Returns the representative points and volumes of the quantized solid.
    """

    def __init__(self, cell_size: float | tuple[float, float, float]) -> None:
        r"""Initializes the quantizer object with the given cell size along each dimension.

        To be implemented.
        """
        raise NotImplementedError

    def get_points_and_volumes(
        self, solid: Solid
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Returns the representative points and volumes of the quantized solid.

        To be implemented.
        """
        raise NotImplementedError


class SphericalQuantizer(Quantizer):
    r"""Abstract class for spherical quantizers.

    A spherical quantizer is an object that quantizes a solid into a set of points and volumes using
    a spherical coordinate system, resulting in a spherical grid.

    Attributes:
        None

    Methods:
        __init__:
            Abstract method that initializes the SphericalQuantizer object.
        get_points_and_volumes:
            Abstract method that returns the representative points and volumes of the quantized
            solid.
        get_surface_points:
            Abstract method that returns the points on the surface of the quantized solid. The
            quantized solid is required to be a sphere.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Abstract method that initializes the SphericalQuantizer object."""

    @abstractmethod
    def get_points_and_volumes(
        self, solid: Solid
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Abstract method, returns the representative points and volumes of the quantized solid.

        Args:
            solid (Solid):
                A Solid object that will be quantized.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            The first array contains the points of the quantized solid, and the second array
            contains the volumes of the quantized cells.
        """

    @abstractmethod
    def get_surface_points(self, solid: Sphere) -> NDArray[np.float32]:
        r"""Abstract method, returns the points on the surface of the quantized sphere.

        Args:
            solid (Sphere):
                A Solid object that will be quantized. It is required to be a sphere.

        Returns (NDArray[np.float32]):
            The points on the surface of the quantized sphere.
        """


class RegularStepsSphericalQuantizer(SphericalQuantizer):
    r"""Quantizer that quantizes a solid on a spherical grid with fixed dimensions.

    Attributes:
        rho_steps_number (int):
            Number of steps along the radius dimension.
        theta_steps_number (int):
            Number of steps along the polar angle dimension.
        phi_steps_number (int):
            Number of steps along the azimuthal angle dimension.

    Methods:
        __init__:
            Initializes the RegularStepsSphericalQuantizer object with the given number of steps
            along each dimension.
        get_points_and_volumes:
            Returns the representative points and volumes of the quantized solid.
        get_surface_points:
            Returns the points on the surface of the quantized sphere.
        spherical_to_cartesian:
            Converts spherical coordinates to cartesian coordinates.
    """

    def __init__(
        self, rho_steps_number: int, theta_steps_number: int, phi_steps_number: int
    ) -> None:
        step_type: str
        step_number: int
        for step_type, step_number in zip(
            ("rho", "theta", "phi"), (rho_steps_number, theta_steps_number, phi_steps_number)
        ):
            if step_number < 1:
                raise ValueError(f"{step_type} step number must be greater than 0")

        self.rho_steps_number: int = rho_steps_number
        self.theta_steps_number: int = theta_steps_number
        self.phi_steps_number: int = phi_steps_number

    def get_points_and_volumes(
        self, solid: Sphere
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Returns the representative points and volumes of the quantized solid.

        Args:
            solid (Solid):
                A Solid object that will be quantized.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            The first array contains the points of the quantized solid, and the second array
            contains the volumes of the quantized cells.
        """
        raise NotImplementedError

        # radius = sphere.radius
        # center = sphere.center

        # min_rho = radius / self.rho_steps_number / 2
        # max_rho = radius - min_rho
        # min_phi = np.pi / self.phi_steps_number / 2
        # max_phi = 2 * np.pi - min_phi

        # rho_values = np.linspace(min_rho, max_rho, self.rho_steps_number - 1, dtype=np.float32)
        # theta_values = np.linspace(0, 2 * np.pi, self.theta_steps_number)
        # phi_values = np.linspace(min_phi, max_phi, self.phi_steps_number - 1)

        # return None  # punti, volumi

    def get_surface_points(self, solid: Sphere) -> NDArray[np.float32]:
        r"""Returns the points on the surface of the quantized sphere.

        Args:
            solid (Sphere):
                A Solid object that will be quantized. It is required to be a sphere.

        Returns (NDArray[np.float32]):
            The points on the surface of the quantized sphere.
        """
        rho: float = solid.radius
        center: NDArray[np.float32] = solid.center

        theta_values: NDArray[np.float32] = np.linspace(
            0,
            2 * np.pi - 2 * np.pi / self.theta_steps_number,
            self.theta_steps_number,
            dtype=np.float32,
        )
        phi_values: NDArray[np.float32] = np.linspace(
            0, np.pi - np.pi / self.phi_steps_number, self.phi_steps_number, dtype=np.float32
        )

        grid: NDArray[np.float32] = np.stack(
            np.meshgrid(theta_values, phi_values), axis=-1
        ).reshape((-1, 2))

        return self.spherical_to_cartesian(rho, grid[:, 0], grid[:, 1], center)

    def spherical_to_cartesian(
        self,
        rho: float,
        theta: NDArray[np.float32],
        phi: NDArray[np.float32],
        center: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        r"""Converts spherical coordinates to cartesian coordinates.

        Args:
            rho (float):
                Radius of the sphere.
            theta (NDArray[np.float32]):
                Polar angle values.
            phi (NDArray[np.float32]):
                Azimuthal angle values.
            center (NDArray[np.float32]):
                Center of the sphere.

        Returns (NDArray[np.float32]):
            The cartesian coordinates of the points.
        """
        x_axis: NDArray[np.float32] = rho * np.sin(phi) * np.cos(theta)
        y_axis: NDArray[np.float32] = rho * np.sin(theta) * np.sin(phi)
        z_axis: NDArray[np.float32] = rho * np.cos(phi)

        return np.stack([x_axis, y_axis, z_axis], axis=-1) + center
