from Solid import Solid

class Sphere(Solid):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def is_inside(self, points):
        return (points - self.center) ** 2 <= self.radius ** 2
        
    def get_surface_points(self, phi_values, theta_values):
        pass

