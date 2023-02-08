from Multisphere import Multisphere
from Sphere import Sphere
from Quantizer import SphericalQuantizer, RegularStepsSphericalQuantizer

from numpy.typing import NDArray
import numpy as np

default_quantizer_class = RegularStepsSphericalQuantizer
default_quantizer_kwargs = {"rho_steps_number": 10, "theta_steps_number": 360, "phi_steps_number": 180}

"""def reduce_multisphere(multisphere: Multisphere, quantizer: SphericalQuantizer):
    max_radii = []


    for candidate in candidate_centers:
        a = max_radius 
        b = max_radius * 2
        
        while b - a > 1:
            sphere = Sphere(candidate, (a + b) / 2)
            points = quantizer.get_surface_points(sphere)
            if protein_multisphere.is_inside_fast(points).all():
                a = (a + b) / 2
            else:
                b = (a + b) / 2

        max_radii.append((a + b) / 2)
        
    print(max_radii)
    print(max(max_radii))"""

def find_candidate_max_radius_points(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float = 1.52,
        multiplier: float = 2,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32)
        ) -> tuple[NDArray[np.float32], float]:

    centers: NDArray[np.float32] = multisphere.get_all_centers()
    max_radius: float = min_radius
    
    quantizer: SphericalQuantizer = default_quantizer_class(**default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

    max_radius_points: list[int] = [0]

    for idx in range(centers.shape[0]):
        if idx in exclude_points:
            continue

        radius: float = min_radius * multiplier
        sphere: Sphere = Sphere(centers[idx], radius)
        points: NDArray[np.float32] = quantizer.get_surface_points(sphere)
        
        while multisphere.is_inside(points)[0].all():
            radius = radius * multiplier
            sphere = Sphere(centers[idx], radius)
            points = quantizer.get_surface_points(sphere)

        last_fitting_radius: float = radius / multiplier

        if last_fitting_radius > max_radius:
            max_radius_points.clear()
            max_radius = last_fitting_radius

        if last_fitting_radius == max_radius:
            max_radius_points.append(idx)

    # candidate_centers = centers[deepest_atoms]

    return np.array(max_radius_points), max_radius

def find_max_radius_point(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float = 1.52,
        multiplier: float = 2,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
        bisection_threshold: float = 1
        ) -> tuple[NDArray[np.int32], np.int32]:
    
    quantizer: SphericalQuantizer = default_quantizer_class(**default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

    candidate_max_radius_points, max_radius = find_candidate_max_radius_points(multisphere, quantizer, min_radius, multiplier=multiplier, exclude_points=exclude_points)
    
    max_radii: NDArray[np.float32] = np.empty(candidate_max_radius_points.shape[0], dtype=np.float32)

    for idx, candidate in enumerate(candidate_max_radius_points):
        a = max_radius 
        b = max_radius * multiplier
        
        while b - a > bisection_threshold:
            sphere = Sphere(candidate, (a + b) / 2)
            points = quantizer.get_surface_points(sphere)
            if multisphere.is_inside(points)[0].all():
                a = (a + b) / 2
            else:
                b = (a + b) / 2

        max_radii[idx] = ((a + b) / 2)

    return candidate_max_radius_points[np.argmax(max_radii)], np.max(max_radii)

def reduce_multisphere_step(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float = 1.52,
        multiplier: float = 2,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
        bisection_threshold: float = 1,
        include_radius: bool = True
        ) -> Multisphere:
    
    quantizer: SphericalQuantizer = default_quantizer_class(**default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

    max_radius_point, max_radius = find_max_radius_point(multisphere, quantizer, min_radius, multiplier, exclude_points, bisection_threshold)

    
    return Multisphere(multisphere.get_all_centers(), multisphere.get_all_radii(), max_radius_point, max_radius)
    

"""def find_reference_radius(multisphere, phi_values, theta_values):
    pass

def find_depth_index(multisphere, point, reference_radius):
    pass

def sadic(protein):
    multisphere = Multisphere(protein)
    reference_radius = find_reference_radius(multisphere)
    
    depth_index_list = []
    for point in multisphere:
        depth_index_list.append(find_depth_index(multisphere, point, reference_radius))"""
