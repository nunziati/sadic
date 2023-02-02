from Solid import Solid
import numpy as np

class Sphere(Solid):
    def __init__(self, center, radius: float):
        if center.shape != (3,):
            raise ValueError("Sphere center must be a 3D vector")
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        self.center = center
        self.radius = radius

    def is_inside(self, points):
        return ((points - self.center) ** 2).sum(axis=1) <= self.radius ** 2
    
    def get_extreme_coordinates(self):
        return np.stack(((self.center - self.radius), (self.center + self.radius)), axis=-1)

    def get_surface_points(self, phi_values=16, theta_values=8):
        pass