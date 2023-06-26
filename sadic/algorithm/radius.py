from numpy.typing import NDArray
import numpy as np

from sadic.solid import Sphere, Multisphere, VoxelSolid
from sadic.quantizer import SphericalQuantizer, RegularStepsSphericalQuantizer

default_quantizer_class = RegularStepsSphericalQuantizer
default_quantizer_kwargs = {"rho_steps_number": 10, "theta_steps_number": 36,
                            "phi_steps_number": 18}

def find_candidate_max_radius_points(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float | None = None,
        multiplier: float = 10000,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32)
        ) -> tuple[NDArray[np.float32], float]:
    
    min_radius = (min_radius if min_radius is not None else
                  multisphere.get_all_radii().min())
    max_radius: float = min_radius

    centers: NDArray[np.float32] = multisphere.get_all_centers()
    
    quantizer: SphericalQuantizer = default_quantizer_class(
        **default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

    max_radius_points: list[int] = [0]

    for idx in range(centers.shape[0]):
        if idx in exclude_points:
            continue

        radius: float = min_radius * multiplier
        sphere: Sphere = Sphere(centers[idx], radius)
        points: NDArray[np.float32] = quantizer.get_surface_points(sphere)
        
        while multisphere.is_inside(points).all():
            radius = radius * multiplier
            sphere = Sphere(centers[idx], radius)
            points = quantizer.get_surface_points(sphere)

        last_fitting_radius: float = radius / multiplier

        if last_fitting_radius > max_radius:
            max_radius_points.clear()
            max_radius = last_fitting_radius

        if last_fitting_radius == max_radius:
            max_radius_points.append(idx)

    return np.array(max_radius_points), max_radius

def find_max_radius_point(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float | None = None,
        multiplier: float = 2,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
        bisection_threshold: float = 0.3
        ) -> tuple[NDArray[np.int32], float]:
    
    min_radius = (multisphere.get_all_radii().min() if min_radius is None else
                  min_radius)

    quantizer: SphericalQuantizer = default_quantizer_class(
        **default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

    candidate_max_radius_points, max_radius = find_candidate_max_radius_points(
        multisphere, quantizer, min_radius, multiplier=multiplier,
        exclude_points=exclude_points)
    
    max_radii: NDArray[np.float32] = np.empty(
        candidate_max_radius_points.shape[0], dtype=np.float32)

    for idx, candidate in enumerate(candidate_max_radius_points):
        candidate_point = multisphere.get_all_centers()[candidate]

        a = max_radius 
        b = max_radius * multiplier
        
        while b - a > bisection_threshold:
            sphere = Sphere(candidate_point, (a + b) / 2)
            points = quantizer.get_surface_points(sphere)

            if multisphere.is_inside(points).all():
                a = (a + b) / 2
            else:
                b = (a + b) / 2

        max_radii[idx] = ((a + b) / 2)

    return (candidate_max_radius_points[np.argmax(max_radii)],
            float(np.max(max_radii)))

def reduce_multisphere_step(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float = 1.52,
        multiplier: float = 2,
        exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
        bisection_threshold: float = 0.3
        ) -> tuple[Multisphere, float, NDArray[np.int32]]:
    
    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs) if quantizer_arg is None else
        quantizer_arg)

    max_radius_point_idx, max_radius = find_max_radius_point(
        multisphere, quantizer, min_radius, multiplier, exclude_points,
        bisection_threshold)
    max_radius_point = multisphere.get_all_centers()[max_radius_point_idx]

    sphere = Sphere(max_radius_point, max_radius)
    points, radii = multisphere.get_all_centers_and_radii()

    is_inside = sphere.is_inside(points, radii)
    points_to_add = np.concatenate((points[np.logical_not(is_inside)],
                                    np.expand_dims(max_radius_point, axis=0)),
                                    axis=0)

    radii_to_add = np.append(radii[np.logical_not(is_inside)], max_radius)
 
    new_multisphere = Multisphere(points_to_add, radii_to_add)
    
    exclude_points = np.arange(
        points_to_add.shape[0] - exclude_points.shape[0] - 1,
        points_to_add.shape[0], dtype=np.int32)
    
    return new_multisphere, max_radius, exclude_points
    

def reduce_multisphere(
        multisphere: Multisphere,
        quantizer_arg: SphericalQuantizer | None = None,
        min_radius: float = 1.52,
        multiplier: float = 2,
        bisection_threshold: float = 0.3
        ) -> Multisphere:

    exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32)
    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs)
        if quantizer_arg is None else quantizer_arg)
    new_multisphere, max_radius, exclude_points = reduce_multisphere_step(
        multisphere, quantizer, min_radius, multiplier, exclude_points,
        bisection_threshold)
    count = 0
    while exclude_points.shape[0] != len(new_multisphere):
        print(count)
        print(max_radius)
        print(exclude_points.shape[0])
        print(len(new_multisphere))
        print("")
        input("Press Enter to continue...")

        count += 1
        new_multisphere, max_radius, exclude_points = reduce_multisphere_step(
            new_multisphere, quantizer, min_radius, multiplier, exclude_points,
            bisection_threshold)

    return new_multisphere

def find_max_radius_point_voxel(voxel: VoxelSolid) -> tuple[int, float]:
    
    centers_indexes = voxel.cartesian_to_grid(
        voxel.multisphere.get_all_centers())

    edt = voxel.edt()
    print("max edt", np.max(edt))
    edt_centers = edt[centers_indexes[:, 0], centers_indexes[:, 1],
                      centers_indexes[:, 2]]
    max_edt_center = np.argmax(edt_centers)
    max_edt = edt_centers[max_edt_center]

    return int(max_edt_center), float(max_edt)