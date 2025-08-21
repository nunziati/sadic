r"""Definition of functions to compute the sadic depth indexes of the atoms of a protein."""

from typing import Type, Union, Optional

from scipy.spatial.distance import cdist

import numpy as np
from numpy.typing import NDArray

from sadic.solid import Sphere, VoxelSolid
from sadic.quantizer import (
    Quantizer,
    RegularStepsSphericalQuantizer,
    RegularStepsCartesianQuantizer,
)
from sadic.solid import Multisphere
from sadic.pdb import Model

default_quantizer_class: Type[Quantizer] = RegularStepsSphericalQuantizer
default_quantizer_kwargs: dict[str, int] = {
    "rho_steps_number": 10,
    "theta_steps_number": 36,
    "phi_steps_number": 18,
}


def sadic_cubes(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Computes the saidc depth indexes of the atoms of a protein.

    For each atom, limit the computation to the atoms within a cube surrounding the probe sphere
    around the atom.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32] = quantizer.get_points_and_volumes(sphere)[0]

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()

    squared_radii: NDArray[np.float32] = (radii**2).reshape((-1, 1)).astype(np.float32)
    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        to_select: NDArray[np.bool_] = (
            (my_centers[:, 0] >= -2.0 - probe_radius)
            & (my_centers[:, 0] <= 2.0 + probe_radius)
            & (my_centers[:, 1] >= -2.0 - probe_radius)
            & (my_centers[:, 1] <= 2.0 + probe_radius)
            & (my_centers[:, 2] >= -2.0 - probe_radius)
            & (my_centers[:, 2] <= 2.0 + probe_radius)
        )
        selected_centers: NDArray[np.float32] = my_centers[to_select]
        selected_radii: NDArray[np.float32] = squared_radii[to_select]

        depth_idx[idx] = (
            2.0
            * (
                ((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(
                    axis=-1
                )
                <= selected_radii
            )
            .any(axis=0)
            .sum()
            / points.shape[0]
        )

    return depth_idx, -1


def sadic_cubes_optimized(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Computes the saidc depth indexes of the atoms of a protein.

    Foreach atom, limit the computation to the atoms within a cube surrounding the probe sphere
    around the atom. Optimized version.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]): The sadic depth indexes of the atoms of the protein
        and the number of atoms with minimum depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32] = quantizer.get_points_and_volumes(sphere)[0]

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()

    squared_radii: NDArray[np.float32] = (radii**2).reshape((-1, 1)).astype(np.float32)
    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)

    to_select: NDArray[np.bool_] = (
        (augmented_centers[:, :, 0] >= -2.0 - probe_radius)
        & (augmented_centers[:, :, 0] <= 2.0 + probe_radius)
        & (augmented_centers[:, :, 1] >= -2.0 - probe_radius)
        & (augmented_centers[:, :, 1] <= 2.0 + probe_radius)
        & (augmented_centers[:, :, 2] >= -2.0 - probe_radius)
        & (augmented_centers[:, :, 2] <= 2.0 + probe_radius)
    )

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        selected_centers: NDArray[np.float32] = my_centers[to_select[idx]]
        selected_radii: NDArray[np.float32] = squared_radii[to_select[idx]]

        depth_idx[idx] = (
            2
            * (
                ((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(
                    axis=-1
                )
                <= selected_radii
            )
            .any(axis=0)
            .sum()
            / points.shape[0]
        )

    return depth_idx, -1


def sadic_sphere(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Computes the saidc depth indexes of the atoms of a protein.

    For each atom, limit the computation to the atoms within a sphere surrounding the probe sphere
    around the atom.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32] = quantizer.get_points_and_volumes(sphere)[0]

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()

    squared_radii: NDArray[np.float32] = (radii**2).reshape((-1, 1)).astype(np.float32)
    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        to_select: NDArray[np.bool_] = (my_centers**2).sum(axis=-1) <= (2.0 + probe_radius) ** 2
        selected_centers: NDArray[np.float32] = my_centers[to_select]
        selected_radii: NDArray[np.float32] = squared_radii[to_select]

        depth_idx[idx] = (
            2
            * (
                ((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(
                    axis=-1
                )
                <= selected_radii
            )
            .any(axis=0)
            .sum()
            / points.shape[0]
        )

    return depth_idx, -1


def sadic_sphere_optimized(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Computes the saidc depth indexes of the atoms of a protein.

    For each atom, limit the computation to the atoms within a sphere surrounding the probe sphere
    around the atom. Optimized version.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32] = quantizer.get_points_and_volumes(sphere)[0]

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()

    squared_radii: NDArray[np.float32] = (radii**2).reshape((-1, 1)).astype(np.float32)
    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)

    to_select: NDArray[np.bool_] = (augmented_centers**2).sum(axis=-1) <= (
        3.5 + probe_radius
    ) ** 2

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        selected_centers: NDArray[np.float32] = my_centers[to_select[idx]]
        selected_radii: NDArray[np.float32] = squared_radii[to_select[idx]]

        depth_idx[idx] = (
            2
            * (
                ((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(
                    axis=-1
                )
                <= selected_radii
            )
            .any(axis=0)
            .sum()
            / points.shape[0]
        )

    return depth_idx, -1


def sadic_norm(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Compute the saidc depth indexes of the atoms of a protein.

    Uses the np.linalg.norm function to compute the distance between the atoms and the points of the
    probe sphere.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32]
    volume: NDArray[np.float32]
    points, volume = quantizer.get_points_and_volumes(sphere)

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    radii = radii.reshape((-1, 1)).astype(np.float32)

    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)
    reference_volume: NDArray[np.float32] = volume * points.shape[0]

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        depth_idx[idx] = (
            2
            / reference_volume
            * (
                np.linalg.norm(
                    points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3)), ord=2, axis=2
                )
                <= radii.astype(np.float32)
            )
            .any(axis=0)
            .sum()
            * volume
        )

    return depth_idx, -1


def sadic_original(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Compute the saidc depth indexes of the atoms of a protein.

    No optimization is used.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere.
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32] = quantizer.get_points_and_volumes(sphere)[0]

    centers: NDArray[np.float32]
    radii: NDArray[np.float32]
    centers, radii = protein_multisphere.get_all_centers_and_radii()

    squared_radii: NDArray[np.float32] = (radii**2).reshape((-1, 1)).astype(np.float32)
    augmented_centers: NDArray[np.float32] = (
        centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
    ).astype(np.float32)

    depth_idx: NDArray[np.float32] = np.empty(centers.shape[0], dtype=np.float32)

    idx: int
    my_centers: NDArray[np.float32]
    for idx, my_centers in enumerate(augmented_centers):
        depth_idx[idx] = (
            2
            * (
                ((points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1)
                <= squared_radii
            )
            .any(axis=0)
            .sum()
            / points.shape[0]
        )

    return depth_idx, -1


def sadic_one_shot(
    protein_multisphere: Multisphere, probe_radius: float, steps_number: Union[int, tuple[int, int, int]]
) -> tuple[NDArray[np.float32], int]:
    r"""Compute the saidc depth indexes of the atoms of a protein.

    Performs the computation in one shot, without using for loops.

    Args:
        protein_multisphere (Multisphere):
            The multisphere describing the protein.
        probe_radius (float):
            The radius of the probe sphere
        steps_number (int | tuple[int, int, int]):
            The number of steps used in the quantization of the probe sphere.

    Returns (tuple[NDArray[np.float32], int]):
        The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
        depth index.
    """
    quantizer: Quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere: Sphere = Sphere(np.array([0.0, 0.0, 0.0]), probe_radius)

    points: NDArray[np.float32]
    volume: NDArray[np.float32]
    points, volume = quantizer.get_points_and_volumes(sphere)

    centers: NDArray[np.float32]
    radii: NDArray[Union[np.float32, np.float16]]
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    radii = radii.reshape((-1, 1)).astype(np.float16)

    augmented_centers: NDArray[Union[np.float32, np.float16]] = (
        (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float16)
    )

    depth_idx: NDArray[np.float32] = (
        np.linalg.norm(
            points.reshape((1, -1, 3)) - augmented_centers.reshape((-1, 1, 3)), ord=2, axis=2
        )
        <= radii
    ).any(axis=0).sum() * volume

    return depth_idx, -1

def cartesian_to_grid(coordinates, extreme_coordinates, resolution):
    return np.floor((coordinates - extreme_coordinates[:, 0]) / resolution).astype(np.int32)

def grid_to_cartesian(coordinates, extreme_coordinates, resolution):
    return coordinates * resolution + extreme_coordinates[:, 0] + resolution / 2

def get_all_coordinate_indexes(grid):
    dimensions = np.array([grid.shape[0], grid.shape[1], grid.shape[2]], dtype=np.int32)
    return (
        np.mgrid[0 : dimensions[0], 0 : dimensions[1], 0 : dimensions[2]]
        .astype(np.int32)
        .transpose(1, 2, 3, 0)
        .reshape(-1, dimensions.shape[0])
    )

def translated_sphere_vectorized(solid, centers, reference_radius, extreme_coordinates, resolution):
    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    sq_reference_radius = reference_radius ** 2
    
    solid_dimensions = np.array([solid.shape[0], solid.shape[1], solid.shape[2]], dtype=np.int32)
    sphere_grid_radius = ((2 * reference_radius / resolution) // 2).astype(np.int32)
    sphere_grid_diameter = sphere_grid_radius * 2 + 1
    grid_origin = np.array([sphere_grid_diameter // 2, sphere_grid_diameter // 2, sphere_grid_diameter // 2], dtype=np.int32)
    sphere_box = np.empty((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter), dtype=np.int32)
    cartesian_origin = grid_to_cartesian(grid_origin, extreme_coordinates, resolution)
    sphere_grid_points = get_all_coordinate_indexes(sphere_box)
    sphere_cartesian_points = grid_to_cartesian(sphere_grid_points, extreme_coordinates, resolution)
    sphere_box = np.where(cdist(sphere_cartesian_points, cartesian_origin.reshape(-1, 3), metric="sqeuclidean") <= sq_reference_radius, 1, 0).reshape((sphere_grid_diameter, sphere_grid_diameter, sphere_grid_diameter))
    sphere_int_volume = np.count_nonzero(sphere_box)

    for idx, center in enumerate(centers):
        grid_center = cartesian_to_grid(center, extreme_coordinates, resolution)

        solid_min_overlapping_coordinates = np.maximum(grid_center - sphere_grid_radius, 0)
        solid_max_overlapping_coordinates = np.minimum(grid_center + sphere_grid_radius + 1, solid_dimensions)
        
        sphere_min_overlapping_coordinates = np.maximum(grid_origin - grid_center, 0)
        sphere_max_overlapping_coordinates = np.minimum(sphere_grid_radius + (solid_dimensions - grid_center), sphere_grid_diameter)
        
        solid_overlap = solid[solid_min_overlapping_coordinates[0]:solid_max_overlapping_coordinates[0], solid_min_overlapping_coordinates[1]:solid_max_overlapping_coordinates[1], solid_min_overlapping_coordinates[2]:solid_max_overlapping_coordinates[2]]
        sphere_overlap = sphere_box[sphere_min_overlapping_coordinates[0]:sphere_max_overlapping_coordinates[0], sphere_min_overlapping_coordinates[1]:sphere_max_overlapping_coordinates[1], sphere_min_overlapping_coordinates[2]:sphere_max_overlapping_coordinates[2]]

        intersection_int_volume = np.count_nonzero(np.logical_and(solid_overlap, sphere_overlap))

        depth_idx[idx] = 2 * (1 - intersection_int_volume / sphere_int_volume)

    return depth_idx


def sadic_original_voxel(
    protein_solid: VoxelSolid, filtered_model: Model, probe_radius: float
) -> tuple[NDArray[np.float32], int]:
    r"""Compute the saidc depth indexes of the atoms of a protein.

    Uses the voxelized solid of the protein.

    Args:
        protein_solid (VoxelSolid):
            The voxelized solid describing the protein.
        filtered_model (Model):
            The model of the protein, filtered to contain only the atoms of interest, on which the
            depth indexes are computed.
        probe_radius (float):
            The radius of the probe sphere.

    Returns:
        (tuple[NDArray[np.float32], int]):
            The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
            depth index.
    """
    centers: NDArray[np.float32] = filtered_model.atoms

    center_number: int = centers.shape[0]

    depth_idx: NDArray[np.float32] = np.zeros(center_number, dtype=np.float32)

    idx: int
    center: NDArray[np.float32]
    for idx, center in enumerate(centers):
        sphere: VoxelSolid = VoxelSolid(
            [Sphere(center, probe_radius)],
            resolution=protein_solid.resolution,
            align_with=protein_solid,
        )
        sphere_volume: int = sphere.int_volume()
        sphere.intersection_(protein_solid)
        intersection_volume: int = sphere.int_volume()
        depth_idx[idx] = 2 * (1 - intersection_volume / sphere_volume)

    count: int = depth_idx[depth_idx == 0.0].shape[0]

    return depth_idx, count

def sadic_fast_voxel(
    protein_solid: VoxelSolid, filtered_model: Model, probe_radius: float
) -> tuple[NDArray[np.float32], int]:
    r"""Compute the saidc depth indexes of the atoms of a protein.

    Uses the voxelized solid of the protein.

    Args:
        protein_solid (VoxelSolid):
            The voxelized solid describing the protein.
        filtered_model (Model):
            The model of the protein, filtered to contain only the atoms of interest, on which the
            depth indexes are computed.
        probe_radius (float):
            The radius of the probe sphere.

    Returns:
        (tuple[NDArray[np.float32], int]):
            The sadic depth indexes of the atoms of the protein and the number of atoms with minimum
            depth index.
    """
    
    centers: NDArray[np.float32] = filtered_model.atoms

    depth_idx = translated_sphere_vectorized(
        protein_solid.grid,
        centers,
        probe_radius,
        protein_solid.extreme_coordinates,
        protein_solid.resolution
    )

    count: int = depth_idx[depth_idx == 0.0].shape[0]

    return depth_idx, count