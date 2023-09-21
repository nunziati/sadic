r"""Definition of functions to compute the max radius sphere in a multisphere-shaped solid."""

from typing import Type

from numpy.typing import NDArray
import numpy as np

from sadic.solid import Sphere, Multisphere, VoxelSolid
from sadic.quantizer import Quantizer, SphericalQuantizer, RegularStepsSphericalQuantizer

default_quantizer_class: Type[Quantizer] = RegularStepsSphericalQuantizer
default_quantizer_kwargs: dict[str, int] = {
    "rho_steps_number": 10,
    "theta_steps_number": 36,
    "phi_steps_number": 18,
}


def find_candidate_max_radius_points(
    multisphere: Multisphere,
    quantizer_arg: SphericalQuantizer | None = None,
    min_radius: float | None = None,
    multiplier: float = 2.0,
    exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
) -> tuple[NDArray[np.float32], float]:
    r"""Finds a set of points among which the generator-center of the maximum inscribed sphere is.

    Finds a set of generating points of a multisphere that are the furthest from the surface of the
    multisphere. The points are found by increasing the radius of the sphere centered at each point
    until the sphere intersects the surface of the multisphere. The radius of the sphere is
    increased by a factor of 'multiplier' each time.

    Args:
        multisphere (Multisphere):
            The multisphere to find the points.
        quantizer_arg (SphericalQuantizer | None, optional):
            A quantizer to use to find the points. When None, a default quantizer is used. Defaults
            to None.
        min_radius (float | None, optional):
            The minimum radius of the spheres to use to find the points. It is used to initialize
            the radius of the growing spheres. When None, the minimum radius of the spheres in the
            multisphere is used. Defaults to None.
        multiplier (float, optional):
            The factor by which the radius of the spheres is increased each time. Defaults to 2.
        exclude_points (NDArray[np.int32], optional):
            The indices of the points to exclude from the search. Defaults to
            np.array([], dtype=np.int32).

    Returns (tuple[NDArray[np.float32], float]):
        The indices of the points that are the furthest from the surface of the multisphere and the
        radius of the maximum tested spheres centered at those points that does not intersect the
        surface of the multisphere.
    """
    min_radius = min_radius if min_radius is not None else multisphere.get_all_radii().min()
    if min_radius is None:
        raise ValueError("The multisphere must contain at least one sphere.")
    
    max_radius: float = min_radius

    centers: NDArray[np.float32] = multisphere.get_all_centers()

    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs)
        if quantizer_arg is None
        else quantizer_arg
    )

    max_radius_points: list[int] = [0]

    idx: int
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
    multiplier: float = 2.0,
    exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
    bisection_threshold: float = 0.3,
) -> tuple[NDArray[np.int32], float]:
    r"""Finds the center of the maximum sphere inscribed in the multisphere, among its generators.

    Finds the generating point of the multisphere that is the furthest from the surface of the
    multisphere. Return the index of the point and the radius of the sphere centered at that point
    that is tangent to the surface of the multisphere.

    Args:
        multisphere (Multisphere):
            The multisphere to find the point.
        quantizer_arg (SphericalQuantizer | None, optional):
            A quantizer to use to find the point. When None, a default quantizer is used. Defaults
            to None.
        min_radius (float | None, optional):
            The minimum radius of the spheres to use to find the point. It is used to initialize
            the radius of the growing spheres. When None, the minimum radius of the spheres in the
            multisphere is used. Defaults to None.
        multiplier (float, optional):
            The factor by which the radius of the spheres is increased each time during the
            preprocessing step that finds the candidate points. Defaults to 2.
        exclude_points (NDArray[np.int32], optional):
            The indices of the points to exclude from the search. Defaults to
            np.array([], dtype=np.int32).
        bisection_threshold (float, optional):
            The threshold used to determine when to stop the bisection search. Defaults to 0.3.

    Returns (tuple[NDArray[np.int32], float]):
        The index of the point that is the furthest from the surface of the multisphere and the
        radius of the sphere centered at that point that is tangent to the surface of the
        multisphere.
    """
    min_radius = multisphere.get_all_radii().min() if min_radius is None else min_radius

    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs)
        if quantizer_arg is None
        else quantizer_arg
    )

    candidate_max_radius_points: NDArray[np.float32]
    max_radius: float
    candidate_max_radius_points, max_radius = find_candidate_max_radius_points(
        multisphere, quantizer, min_radius, multiplier=multiplier, exclude_points=exclude_points
    )

    max_radii: NDArray[np.float32] = np.empty(
        candidate_max_radius_points.shape[0], dtype=np.float32
    )

    idx: int
    candidate: int
    for idx, candidate in enumerate(candidate_max_radius_points):
        candidate_point = multisphere.get_all_centers()[candidate]

        left_extreme: float = max_radius
        right_extreme: float = max_radius * multiplier

        while right_extreme - left_extreme > bisection_threshold:
            sphere: Sphere = Sphere(candidate_point, (left_extreme + right_extreme) / 2)
            points: NDArray[np.float32] = quantizer.get_surface_points(sphere)

            if multisphere.is_inside(points).all():
                left_extreme = (left_extreme + right_extreme) / 2
            else:
                right_extreme = (left_extreme + right_extreme) / 2

        max_radii[idx] = (left_extreme + right_extreme) / 2

    return (candidate_max_radius_points[np.argmax(max_radii)], float(np.max(max_radii)))


def reduce_multisphere_step(
    multisphere: Multisphere,
    quantizer_arg: SphericalQuantizer | None = None,
    min_radius: float | None = None,
    multiplier: float = 2.0,
    exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32),
    bisection_threshold: float = 0.3,
) -> tuple[Multisphere, float, NDArray[np.int32]]:
    r"""Perform a step of multisphere reduction.

    Replaces a set of spheres composing the multisphere with a single sphere that surrounds them and
    does not intersect with the surface of the multisphere.

    Args:
        multisphere (Multisphere):
            The multisphere to reduce.
        quantizer_arg (SphericalQuantizer | None, optional):
            A quantizer to use to find the points to be replaced. When None, a default quantizer is
            used. Defaults to None.
        min_radius (float | None, optional):
            The minimum radius of the spheres to use to find the point. It is used to initialize
            the radius of the growing spheres. When None, the minimum radius of the spheres in the
            multisphere is used. Defaults to None.
        multiplier (float, optional):
            The factor by which the radius of the spheres is increased each time during the
            preprocessing step that finds the candidate points. Defaults to 2.
        exclude_points (NDArray[np.int32], optional):
            The indices of the points to exclude from the reduction. Defaults to
            np.array([], dtype=np.int32).
        bisection_threshold (float, optional):
            The threshold used to determine when to stop the bisection search. Defaults to 0.3.

    Returns (tuple[Multisphere, float, NDArray[np.int32]]):
        The multisphere reduced by a step, the radius of the sphere that surrounds the replaced
        spheres and the indices of the replaced spheres.
    """
    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs)
        if quantizer_arg is None
        else quantizer_arg
    )

    max_radius_point_idx: NDArray[np.int32]
    max_radius: float
    max_radius_point_idx, max_radius = find_max_radius_point(
        multisphere, quantizer, min_radius, multiplier, exclude_points, bisection_threshold
    )

    max_radius_point: NDArray[np.float32]
    max_radius_point = multisphere.get_all_centers()[max_radius_point_idx]

    sphere: Sphere = Sphere(max_radius_point, max_radius)
    points: NDArray[np.float32]
    radii: NDArray[np.float32]
    points, radii = multisphere.get_all_centers_and_radii()

    is_inside: NDArray[np.bool_] = sphere.is_inside(points, radii)
    points_to_add: NDArray[np.float32] = np.concatenate(
        (points[np.logical_not(is_inside)], np.expand_dims(max_radius_point, axis=0)), axis=0
    )

    radii_to_add: NDArray[np.float32] = np.append(radii[np.logical_not(is_inside)], max_radius)

    new_multisphere: Multisphere = Multisphere(points_to_add, radii_to_add)

    exclude_points = np.arange(
        points_to_add.shape[0] - exclude_points.shape[0] - 1, points_to_add.shape[0], dtype=np.int32
    )

    return new_multisphere, max_radius, exclude_points


def reduce_multisphere(
    multisphere: Multisphere,
    quantizer_arg: SphericalQuantizer | None = None,
    min_radius: float | None = None,
    multiplier: float = 2.0,
    bisection_threshold: float = 0.3,
) -> Multisphere:
    r"""Reduce multisphere by applying a series of steps of multisphere reduction.

    The reduction stops when the multisphere cannot be reduced anymore.

    Args:
        multisphere (Multisphere):
            The multisphere to reduce.
        quantizer_arg (SphericalQuantizer | None, optional):
            A quantizer to use to find the points to be replaced. When None, a default quantizer is
            used. Defaults to None.
        min_radius (float | None, optional):
            The minimum radius of the spheres to use to find the point. It is used to initialize the
            radius of the growing spheres. When None, the minimum radius of the spheres in the
            multisphere is used. Defaults to None.
        multiplier (float, optional):
            The factor by which the radius of the spheres is increased each time during the
            preprocessing step that finds the candidate points. Defaults to 2.
        bisection_threshold (float, optional):
            The threshold used to determine when to stop the bisection search. Defaults to 0.3.

    Returns (Multisphere):
        The reduced multisphere.
    """
    exclude_points: NDArray[np.int32] = np.array([], dtype=np.int32)
    quantizer: SphericalQuantizer = (
        default_quantizer_class(**default_quantizer_kwargs)
        if quantizer_arg is None
        else quantizer_arg
    )
    new_multisphere: Multisphere
    max_radius: float
    exclude_points: NDArray[np.int32]
    new_multisphere, max_radius, exclude_points = reduce_multisphere_step(
        multisphere, quantizer, min_radius, multiplier, exclude_points, bisection_threshold
    )

    count: int = 0
    while exclude_points.shape[0] != len(new_multisphere):
        print(count)
        print(max_radius)
        print(exclude_points.shape[0])
        print(len(new_multisphere))
        print("")
        input("Press Enter to continue...")

        count += 1
        new_multisphere, max_radius, exclude_points = reduce_multisphere_step(
            new_multisphere, quantizer, min_radius, multiplier, exclude_points, bisection_threshold
        )

    return new_multisphere


def find_max_radius_point_voxel(voxel: VoxelSolid) -> tuple[int, float]:
    r"""Finds the maximum inscribed sphere in the voxel solid that has a generating point as center.

    Finds the generating point of the voxel representation of the multisphere that is the furthest
    from the surface of the voxel solid. Return the index of the point and the radius of the sphere
    centered at that point that is tangent to the surface of the voxel solid.

    Args:
        voxel (VoxelSolid):
            The voxel solid describing the multisphere.

    Returns (tuple[int, float]):
        The index of the point and the radius of the sphere centered at that point that is tangent
        to the surface of the voxel solid.
    """
    centers_indexes: NDArray[np.int32] = voxel.cartesian_to_grid(
        voxel.multisphere.get_all_centers()
    )

    edt: NDArray[np.float32] = voxel.edt()
    print("max edt", np.max(edt))
    edt_centers: NDArray[np.float32] = edt[
        centers_indexes[:, 0], centers_indexes[:, 1], centers_indexes[:, 2]
    ]
    max_edt_center: int = int(np.argmax(edt_centers))
    max_edt: float = edt_centers[max_edt_center]

    return max_edt_center, max_edt
