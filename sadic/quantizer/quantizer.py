from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sadic.solid import Solid, Sphere

class Quantizer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_points_and_volumes(
            self,
            solid) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        pass

    def get_extreme_coordinates(self, solid: Solid):
        if hasattr(solid, "get_extreme_coordinates"):
            return solid.get_extreme_coordinates()
        else:
            raise NotImplementedError

class CartesianQuantizer(Quantizer):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_points_and_volumes(
            self,
            arg: Solid) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        pass

class RegularStepsCartesianQuantizer(CartesianQuantizer):
    def __init__(self, steps_number: int | tuple[int, int, int]) -> None:
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
                ("x", "y", "z"),
                (x_steps_number, y_steps_number, z_steps_number)):
            if step_number < 1:
                raise ValueError(
                    f"{step_type} step number must be greater than 0")

        self.x_steps_number: int = x_steps_number
        self.y_steps_number: int = y_steps_number
        self.z_steps_number: int = z_steps_number

    def get_points_and_volumes(
            self,
            solid: Solid) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        extremes: NDArray[np.float32] = self.get_extreme_coordinates(solid)
        x_values: NDArray[np.float32] = np.linspace(extremes[0][0],
                                                    extremes[0][1],
                                                    self.x_steps_number,
                                                    dtype=np.float32)
        y_values: NDArray[np.float32] = np.linspace(extremes[1][0],
                                                    extremes[1][1],
                                                    self.y_steps_number,
                                                    dtype=np.float32)
        z_values: NDArray[np.float32] = np.linspace(extremes[2][0],
                                                    extremes[2][1],
                                                    self.z_steps_number,
                                                    dtype=np.float32)

        volume: NDArray[np.float32] = ((x_values[1] - x_values[0])
            * (y_values[1] - y_values[0])
            * (z_values[1] - z_values[0]))
        
        grid: NDArray[np.float32] = np.stack(np.meshgrid(x_values, y_values, z_values),
                        axis=-1).reshape((-1, 3))

        grid_inside: NDArray[np.float32] = grid[solid.is_inside(grid)]

        return (grid_inside,
                np.full((grid_inside.shape[0],), volume, dtype=np.float32))


class RegularSizeCartesianQuantizer(RegularStepsCartesianQuantizer):
    def __init__(self) -> None:
        pass
    
    def get_points_and_volumes(self): # -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        pass

class SphericalQuantizer(Quantizer):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_points_and_volumes(
            self,
            *args) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        pass

    @abstractmethod
    def get_surface_points(self, solid: Sphere) -> NDArray[np.float32]:
        pass

class RegularStepsSphericalQuantizer(SphericalQuantizer):
    def __init__(
            self,
            rho_steps_number: int,
            theta_steps_number: int,
            phi_steps_number: int) -> None:
        step_type: str
        step_number: int          
        for step_type, step_number in zip(
                ("rho", "theta", "phi"),
                (rho_steps_number, theta_steps_number, phi_steps_number)):
            if step_number < 1:
                raise ValueError(
                    f"{step_type} step number must be greater than 0")

        self.rho_steps_number: int = rho_steps_number
        self.theta_steps_number: int = theta_steps_number
        self.phi_steps_number: int = phi_steps_number

    def get_points_and_volumes(self, sphere: Sphere) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        raise NotImplementedError
    
        radius = sphere.radius
        center = sphere.center
        
        min_rho = radius / self.rho_steps_number / 2
        max_rho = radius - min_rho
        min_phi = np.pi / self.phi_steps_number / 2
        max_phi = 2 * np.pi - min_phi

        rho_values = np.linspace(min_rho, max_rho,
                                 self.rho_steps_number - 1, dtype=np.float32)
        theta_values = np.linspace(0, 2 * np.pi, self.theta_steps_number)
        phi_values = np.linspace(min_phi, max_phi, self.phi_steps_number - 1)

        return None # punti, volumi

    def get_surface_points(self, solid: Sphere) -> NDArray[np.float32]:
        rho: float = solid.radius
        center: NDArray[np.float32] = solid.center
        
        theta_values: NDArray[np.float32] = np.linspace(
            0, 2 * np.pi - 2 * np.pi / self.theta_steps_number,
            self.theta_steps_number, dtype=np.float32)
        phi_values: NDArray[np.float32] = np.linspace(
            0, np.pi - np.pi / self.phi_steps_number, self.phi_steps_number,
            dtype=np.float32)

        grid: NDArray[np.float32] = np.stack(
            np.meshgrid(theta_values, phi_values), axis=-1).reshape((-1, 2))

        return self.spherical_to_cartesian(rho, grid[:, 0], grid[:, 1], center)

    def spherical_to_cartesian(
            self,
            rho,
            theta,
            phi,
            center) -> NDArray[np.float32]:
        x: NDArray[np.float32] = rho * np.sin(phi) * np.cos(theta)
        y: NDArray[np.float32] = rho * np.sin(theta) * np.sin(phi)
        z: NDArray[np.float32] = rho * np.cos(phi)

        return np.stack([x, y, z], axis=-1) + center

class RegularSizeSphericalQuantizer(SphericalQuantizer):
    def __init__(self) -> None:
        raise NotImplementedError
        pass

    def get_points(self, *args) -> NDArray[np.float32]:
        raise NotImplementedError
        pass

class AdaptiveSphericalQuantizer(SphericalQuantizer):
    def __init__(self) -> None:
        raise NotImplementedError
        pass

    def get_points(self, *args) -> NDArray[np.float32]:
        raise NotImplementedError
        pass

class SamplerQuantizer(Quantizer):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_points(self, *args) -> NDArray[np.float32]:
        pass

class SphereSamplerQuantizer(SamplerQuantizer):
    def __init__(self) -> None:
        raise NotImplementedError
        pass

    def get_points(self, *args) -> NDArray[np.float32]:
        raise NotImplementedError
        pass