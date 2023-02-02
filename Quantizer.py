from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from Sphere import Sphere
from Solid import Solid

class Quantizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_points_and_volumes(self, x):
        pass

class CartesianQuantizer(Quantizer):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_points_and_volumes(self, *args) -> tuple[NDArray[np.float32], float]:
        pass

class RegularStepsCartesianQuantizer(CartesianQuantizer):
    def __init__(self, steps_number: int | tuple[int, int, int]):
        if isinstance(steps_number, int):
            x_steps_number, y_steps_number, z_steps_number = steps_number, steps_number, steps_number
        else:
            x_steps_number, y_steps_number, z_steps_number = steps_number
            
        for step_type, step_number in zip(("x", "y", "z"), (x_steps_number, y_steps_number, z_steps_number)):
            if step_number < 1:
                raise ValueError(f"{step_type} step number must be greater than 0")
                
        super().__init__()

        self.x_steps_number = x_steps_number
        self.y_steps_number = y_steps_number
        self.z_steps_number = z_steps_number

    def get_extreme_coordinates(self, solid: Solid):
        # check if solid has method get_extreme_coordinates
        if hasattr(solid, "get_extreme_coordinates"):
            return solid.get_extreme_coordinates()
        else:
            raise NotImplementedError

    def get_points_and_volumes(self, solid: Solid) -> tuple[NDArray[np.float32], float]:
        extremes = self.get_extreme_coordinates(solid)
        x_values = np.linspace(extremes[0][0], extremes[0][1], self.x_steps_number, dtype=np.float32)
        y_values = np.linspace(extremes[1][0], extremes[1][1], self.y_steps_number, dtype=np.float32)
        z_values = np.linspace(extremes[2][0], extremes[2][1], self.z_steps_number, dtype=np.float32)

        volume = (x_values[1] - x_values[0]) * (y_values[1] - y_values[0]) * (z_values[1] - z_values[0])
        grid = np.stack(np.meshgrid(x_values, y_values, z_values), axis=-1).reshape((-1, 3))

        return grid[solid.is_inside(grid)], volume



class RegularSizeCartesianQuantizer(CartesianQuantizer):
    def __init__(self, ):
        super().__init__()

    def get_points_and_volumes(self):
        pass

class SphericalQuantizer(Quantizer):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_points_and_volumes(self, *args):
        pass

class RegularStepsSphericalQuantizer(SphericalQuantizer):
    def __init__(self, rho_steps_number: int, theta_steps_number: int, phi_steps_number: int):            
        for step_type, step_number in zip(("rho", "theta", "phi"), (rho_steps_number, theta_steps_number, phi_steps_number)):
            if step_number < 1:
                raise ValueError(f"{step_type} step number must be greater than 0")
                
        super().__init__()

        self.rho_steps_number = rho_steps_number
        self.theta_steps_number = theta_steps_number
        self.phi_steps_number = phi_steps_number

    def get_points_and_volumes(self, sphere: Sphere):
        radius = sphere.radius
        center = sphere.center
        
        min_rho = radius / self.rho_steps_number / 2
        max_rho = radius - min_rho
        min_phi = np.pi / self.phi_steps_number / 2
        max_phi = 2 * np.pi - min_phi

        rho_values = np.linspace(min_rho, max_rho, self.rho_steps_number - 1, dtype=np.float32)
        theta_values = np.linspace(0, 2 * np.pi, self.theta_steps_number)
        phi_values = np.linspace(min_phi, max_phi, self.phi_steps_number - 1)



class RegularSizeSphericalQuantizer(SphericalQuantizer):
    def __init__(self):
        super().__init__()

    def get_points(self, *args):
        pass

class AdaptiveSphericalQuantizer(SphericalQuantizer):
    def __init__(self):
        super().__init__()

    def get_points(self, *args):
        pass

class SamplerQuantizer(Quantizer):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_points(self, *args):
        pass

class SphereSamplerQuantizer(SamplerQuantizer):
    def __init__(self):
        super().__init__()

    def get_points(self, *args):
        pass