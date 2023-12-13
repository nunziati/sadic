r"""Defines the VoxelSolid class."""

from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Type

from tqdm import tqdm
from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt

from sadic.solid import Solid, Sphere, Multisphere
from sadic.pdb import Model
from sadic.quantizer import Quantizer, RegularStepsCartesianQuantizer


class VoxelSolid(Solid):
    r"""A solid defined by a 3D grid of voxels. It represents a multisphere.

    Attributes:
        multisphere (Multisphere):
            The multisphere that defines the solid.
        resolution (float):
            The resolution of the grid.
        extreme_coordinates (NDArray[np.float32]):
            The extreme coordinates of the grid in the cartesian space.
        dimensions (NDArray[np.int32]):
            The dimensions of the grid, in voxels.
        grid (NDArray[np.bool_]):
            The grid of voxels.

    Methods:
        __init__:
            Initializes the VoxelSolid.
        build_from_multisphere:
            Builds the VoxelSolid from a Multisphere.
        get_all_grid_coordinates:
            Returns the coordinates of all the voxels in the grid, in indexes.
        cartesian_to_grid:
            Converts cartesian coordinates to grid indexes.
        grid_to_cartesian:
            Converts grid indexes to cartesian coordinates.
        get_extreme_coordinates:
            Returns the extreme coordinates of the grid in the cartesian space.
        remove_holes_:
            Removes holes from the solid inplace.
        remove_holes:
            Removes holes from the solid and returns a new solid.
        edt:
            Returns the Euclidean Distance Transform of the solid.
        translate:
            Translates the solid and returns a new solid.
        is_inside:
            Method to check if a (set of) point(s) or a sphere is inside the solid.
        point_is_inside:
            Method to check if a (set of) point(s) is inside the solid.
        sphere_is_inside:
            Method to check if a sphere is inside the solid.
        local_function:
            Finds the intersection of two voxel solids and applies a given function to it.
        local_operator:
            Applies a function that operates on the intersection of two voxel solids and writes the
            result in the first one.
        intersection_:
            Finds the intersection of two voxel solids and writes the result in the first one.
        intersection:
            Finds the intersection of two voxel solids and returns it as a new solid.
        union_:
            Finds the union of two voxel solids and writes the result in the first one.
        union:
            Finds the union of two voxel solids and returns it as a new solid.
        voxel_volume:
            Returns the volume of a voxel.
        volume:
            Returns the volume of the solid.
        int_volume:
            Returns the volume of the solid in the integer space i.e. the number of voxels composing
            the solid.
    """
    default_quantizer_class: Type[Quantizer] = RegularStepsCartesianQuantizer
    default_quantizer_kwargs: dict[str, int] = {"steps_number": 32}

    def __init__(
        self,
        arg1: (Sequence[Sphere] | Model | Structure | NDArray[np.float32]),
        arg2: None | NDArray[np.float32] = None,
        resolution: float = 0.3,
        extreme_coordinates: None | NDArray[np.float32] = None,
        align_with: None | VoxelSolid = None,
    ) -> None:
        r"""Initializes the VoxelSolid building it from a given argument.

        The argument can be of different types and the VoxelSolid is built accordingly. The
        constructor offers the possibility of choosing the resolution of the grid and the extreme
        coordinates of the solid. The built solid can also be aligned with another solid.

        Args:
            arg1 (Sequence[Sphere] | Model | Structure | NDArray[np.float32]):
                The argument from which to build the VoxelSolid. It can be a sequence of spheres,
                a sadic.Model object, a BioPython Structure object or a numpy array of shape (N, 3)
                containing the cartesian coordinates of the centers of the spheres.
            arg2 (None | NDArray[np.float32]):
                If arg1 is a numpy array, arg2 must be a numpy array of shape (N,) containing the
                radii of the spheres.
            resolution (float):
                The resolution of the grid in the cartesian space.
            extreme_coordinates (None | NDArray[np.float32]):
                The extreme coordinates of the grid in the cartesian space. If None, they are
                computed from the extreme coordinates of the multisphere that defines the solid.
            align_with (None | VoxelSolid):
                If not None, the solid is aligned with the given solid, in a way that the two grids
                have the same resolution and the difference between the extreme coordinates of the
                two grids is a multiple of the resolution.
        """
        self.multisphere: Multisphere
        self.resolution: float
        self.extreme_coordinates: NDArray[np.float32]
        self.dimensions: NDArray[np.int32]
        self.grid: NDArray[np.bool_]

        if isinstance(arg1, Multisphere):
            self.build_from_multisphere(arg1, resolution, extreme_coordinates=extreme_coordinates)
        else:
            multisphere: Multisphere = Multisphere(arg1, arg2)
            self.build_from_multisphere(
                multisphere,
                resolution,
                extreme_coordinates=extreme_coordinates,
                align_with=align_with,
            )

    def build_from_multisphere(
        self,
        multisphere: Multisphere,
        resolution: float = 0.3,
        extreme_coordinates: None | NDArray[np.float32] = None,
        align_with: "None | VoxelSolid" = None,
    ) -> None:
        r"""Builds the VoxelSolid from a Multisphere.

        The constructor offers the possibility of choosing the resolution of the grid and the
        extreme coordinates of the solid. The built solid can also be aligned with another solid.

        Args:
            multisphere (Multisphere):
                The multisphere from which to build the VoxelSolid.
            resolution (float):
                The resolution of the grid in the cartesian space.
            extreme_coordinates (None | NDArray[np.float32]):
                The extreme coordinates of the grid in the cartesian space. If None, they are
                computed from the extreme coordinates of the multisphere that defines the solid.
            align_with (None | VoxelSolid):
                If not None, the solid is aligned with the given solid, in a way that the two grids
                have the same resolution and the difference between the extreme coordinates of the
                two grids is a multiple of the resolution.
        """
        self.multisphere = multisphere
        self.resolution = resolution

        if extreme_coordinates is None:
            self.extreme_coordinates = multisphere.get_extreme_coordinates()
            self.extreme_coordinates[:, 1] += (
                self.resolution
                * (1 - np.modf(
                    (self.extreme_coordinates[:, 1] - self.extreme_coordinates[:, 0])
                    / self.resolution
                )[0])
            )
            if align_with is not None:
                if self.resolution != align_with.resolution:
                    raise ValueError("resolution must be the same as align_with.resolution")
                grid_offset: NDArray[np.float32] = (
                    align_with.extreme_coordinates[:, 0] - self.extreme_coordinates[:, 0]
                ) / self.resolution
                self.extreme_coordinates = self.extreme_coordinates + (
                    (grid_offset - np.round(grid_offset)) * self.resolution
                ).reshape(-1, 1)
        else:
            self.extreme_coordinates = extreme_coordinates

        self.dimensions = np.ceil(
            (self.extreme_coordinates[:, 1] - self.extreme_coordinates[:, 0]) / self.resolution
        ).astype(np.int32)
        self.grid = np.full(self.dimensions, False, dtype=np.bool_)

        if len(self.multisphere) == 1:
            self.grid = multisphere.is_inside(
                self.grid_to_cartesian(self.get_all_grid_coordinates())
            ).reshape(self.dimensions)
        else:
            center: NDArray[np.float32]
            radius: float
            for center, radius in tqdm(zip(*self.multisphere.get_all_centers_and_radii())):
                sphere = VoxelSolid(
                    [Sphere(center, radius)], resolution=self.resolution, align_with=self
                )
                self.union_(sphere)

    def get_all_grid_coordinates(self) -> NDArray[np.int32]:
        r"""Returns an array containing all the coordinates of the grid.

        Returns (NDArray[np.int32]):
            An array of shape (N, 3) containing all the coordinates of the grid in the integer
            space.
        """
        return (
            np.mgrid[0 : self.dimensions[0], 0 : self.dimensions[1], 0 : self.dimensions[2]]
            .astype(np.int32)
            .transpose(1, 2, 3, 0)
            .reshape(-1, self.dimensions.shape[0])
        )

    def cartesian_to_grid(self, coordinates: NDArray[np.float32]) -> NDArray[np.int32]:
        r"""Converts cartesian coordinates to grid coordinates in the space of the solid.

        Args:
            coordinates (NDArray[np.float32]):
                An array of shape (N, 3) containing the cartesian coordinates to convert.

        Returns (NDArray[np.int32]):
            An array of shape (N, 3) containing the grid coordinates.
        """
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D coordinates")

        return np.floor((coordinates - self.extreme_coordinates[:, 0]) / self.resolution).astype(
            np.int32
        )

    def grid_to_cartesian(self, coordinates: NDArray[np.int32]) -> NDArray[np.float32]:
        r"""Converts grid coordinates in the space of the solid to cartesian coordinates.

        Args:
            coordinates (NDArray[np.int32]):
                An array of shape (N, 3) containing the grid coordinates to convert.

        Returns (NDArray[np.float32]):
            An array of shape (N, 3) containing the cartesian coordinates.
        """
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D indexes")

        return coordinates * self.resolution + self.extreme_coordinates[:, 0] + self.resolution / 2

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        r"""Returns the extreme coordinates of the solid in the cartesian space.

        Returns (NDArray[np.float32]):
            An array of shape (3, 2) containing the extreme coordinates of the solid in the
            cartesian space.
        """
        return self.extreme_coordinates

    def remove_holes_(self) -> None:
        r"""Removes the holes in the solid inplace.

        Computes the connected components of the solid and removes the components that are not
        connected to the main component.
        """
        self.grid = label(self.grid.astype(np.int32))[0] != 0

    def remove_holes(self, *args, **kwargs) -> VoxelSolid:
        r"""Removes the holes in the solid and returns a new solid.

        Computes the connected components of the solid and removes the components that are not
        connected to the main component.

        Returns (VoxelSolid):
            A new solid without holes.
        """
        new_voxel_solid: VoxelSolid = deepcopy(self)
        new_voxel_solid.remove_holes_(*args, **kwargs)
        return new_voxel_solid

    def edt(self, sampling: None | float = None) -> NDArray[np.float32]:
        r"""Computes the euclidean distance transform of the solid.

        Returns (NDArray[np.float32]):
            An array of the same shape as the grid, containing the euclidean distance transform of
            the solid.
        """

        if sampling is None:
            sampling = self.resolution
            
        return distance_transform_edt(self.grid, sampling=sampling)

    def translate(self, shift: NDArray[np.int32]) -> VoxelSolid:
        r"""Translates the solid by the given shift in the grid space and returns a new solid.

        Args:
            shift (NDArray[np.int32]):
                An array of shape (3,) containing the shift to apply to the solid in the grid
                space.

        Returns (VoxelSolid):
            A new solid translated by the given shift.
        """
        if shift.ndim != 1 or shift.shape[0] != 3:
            raise ValueError("shift must be a 1D array of 3 elements")

        new_voxel_solid: VoxelSolid = deepcopy(self)

        new_voxel_solid.grid = np.roll(new_voxel_solid.grid, shift, axis=(0, 1, 2))

        axis: int
        for axis in range(3):
            array_slice: list[slice] = [slice(None)] * new_voxel_solid.grid.ndim
            if shift[axis] > 0:
                array_slice[axis] = slice(0, shift[axis])
                new_voxel_solid.grid[tuple(array_slice)] = False
            else:
                array_slice[axis] = slice(shift[axis], None)
                new_voxel_solid.grid[tuple(array_slice)] = False

        return new_voxel_solid

    def is_inside(self, *args, **kwargs) -> NDArray[np.bool_]:
        r"""Checks if a (set of) point(s) or a sphere is inside the solid.

        Automatically detects if the argument is a point or a sphere and calls the appropriate
        method. The method can also return the volumes of the quantized cells containing the points
        of the sphere.

        Args:
            arg (NDArray[np.float32] | Sphere):
                The point(s) or sphere to check.
            get_volumes (bool, optional):
                Whether to return the volumes of the quantized cells containing the points of the
                sphere. Defaults to False.

        Returns (NDArray[np.bool_]):
            A numpy.ndarray object of shape (n,) containing the result of the check for each point
            or for each representative point of the sphere.
        """
        # TO DO: implement the get_volume option
        arg: NDArray[np.float32] | Sphere = args[0]
        get_volumes: bool = kwargs.get("get_volumes", False)
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg, get_volumes=get_volumes)

        elif isinstance(arg, np.ndarray):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a numpy.ndarray or a Sphere")

    def point_is_inside_boundaries(self, points: NDArray[np.float32]) -> NDArray[np.bool_]:
        r"""Checks if a (set of) point(s) is inside the grid boundaries.

        Args:
            points (NDArray[np.float32]):
                The point(s) to check.

        Returns (NDArray[np.bool_]):
            A numpy.ndarray object of shape (n,) containing the result of the check for each point.
        """
        return np.logical_and(
            np.logical_and(
                np.logical_and(
                    points[:, 0] > self.extreme_coordinates[0, 0],
                    points[:, 0] < self.extreme_coordinates[0, 1],
                ),
                np.logical_and(
                    points[:, 1] > self.extreme_coordinates[1, 0],
                    points[:, 1] < self.extreme_coordinates[1, 1],
                ),
            ),
            np.logical_and(
                points[:, 2] > self.extreme_coordinates[2, 0],
                points[:, 2] < self.extreme_coordinates[2, 1],
            ),
        )

    def point_is_inside(self, points: NDArray[np.float32]) -> NDArray[np.bool_]:
        r"""Checks if a set of points is inside the solid.

        Args:
            points (NDArray[np.float32]):
                The points to check.

        Returns (NDArray[np.bool_]):
            A numpy.ndarray object of shape (n,) containing the result of the check for each point.
        """
        output: NDArray[np.bool_] = self.point_is_inside_boundaries(points)
        grid_points: NDArray[np.int32] = self.cartesian_to_grid(points[output])
        output[output] = self.grid[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]]
        return output

    def sphere_is_inside(
        self, sphere: Sphere, quantizer_arg: Quantizer | None = None, get_volumes: bool = False
    ) -> NDArray[np.bool_]:
        r"""Checks if a sphere is inside the multisphere.

        Quantizes the sphere and checks if the quantized points are inside the solid. The method can
        also return the volumes of the quantized cells containing the points of the sphere.

        Args:
            sphere (Sphere):
                The sphere to check.
            quantizer_arg (Quantizer, optional):
                The quantizer to use. When None, the default quantizer is used. Defaults to None.
            get_volumes (bool, optional):
                Whether to return the volumes of the quantized cells containing the points of the
                sphere. Defaults to False.

        Returns (NDArray[np.bool_]):
            A numpy.ndarray object of shape (n,) containing the result of the check for each point
            of the sphere.
        """
        quantizer: Quantizer

        if quantizer_arg is None:
            quantizer = Multisphere.default_quantizer_class(**Multisphere.default_quantizer_kwargs)
        else:
            quantizer = quantizer_arg

        points: NDArray[np.float32]
        points, _ = quantizer.get_points_and_volumes(sphere)

        if get_volumes:
            return self.point_is_inside(points)

        return self.point_is_inside(points)

    def local_function(
        self,
        function: Callable[[VoxelSolid, VoxelSolid, NDArray[np.int32], NDArray[np.int32]], None],
        other: VoxelSolid,
    ) -> None:
        r"""Finds the intersection of two voxel solids and applies a given function to it.

        Args:
            function (Callable[[VoxelSolid, VoxelSolid, NDArray[np.int32], NDArray[np.int32]],
            None]):
                The function to apply.
            other (VoxelSolid):
                The other solid.
        """
        # check if grids are aligned
        if self.resolution != other.resolution:
            raise ValueError("Grids must have the same resolution")
        displacement: NDArray[np.float32] = (
            self.extreme_coordinates - other.extreme_coordinates
        ) / self.resolution
        if np.any(np.round(displacement - np.round(displacement, decimals=2), decimals=2)[:, 0]):
            raise ValueError("Grids must be aligned")

        min_intersection_centers: NDArray[np.float32] = (
            np.max(
                np.stack((self.extreme_coordinates[:, 0], other.extreme_coordinates[:, 0]), axis=1),
                axis=1,
            )
            + self.resolution / 2
        )
        max_intersection_centers: NDArray[np.float32] = (
            np.min(
                np.stack((self.extreme_coordinates[:, 1], other.extreme_coordinates[:, 1]), axis=1),
                axis=1,
            )
            + self.resolution / 2
        )

        self_intersection_extremes: NDArray[np.int32] = np.stack(
            (
                self.cartesian_to_grid(min_intersection_centers),
                self.cartesian_to_grid(max_intersection_centers),
            ),
            axis=1,
        )
        other_intersection_extremes: NDArray[np.int32] = np.stack(
            (
                other.cartesian_to_grid(min_intersection_centers),
                other.cartesian_to_grid(max_intersection_centers),
            ),
            axis=1,
        )

        function(self, other, self_intersection_extremes, other_intersection_extremes)

    def local_operator(
        self,
        operator: Callable[[NDArray[np.bool_], NDArray[np.bool_]], NDArray[np.bool_]],
        other: VoxelSolid,
        default = None,
    ) -> None:
        r"""Applies a function that operates on the intersection of two voxel solids inplace.

        Writes the result in the first one.

        Args:
            operator (Callable[[NDArray[np.bool_], NDArray[np.bool_]], NDArray[np.bool_]]):
                The function to apply.
            other (VoxelSolid):
                The other solid.
        """

        def function(
            self: VoxelSolid,
            other: VoxelSolid,
            self_intersection_extremes: NDArray[np.int32],
            other_intersection_extremes: NDArray[np.int32],
        ) -> None:
            grid_copy: NDArray[np.bool_] = np.array([])
            if default is not None:
                grid_copy = self.grid.copy()
                self.grid[:, :, :] = default
            self.grid[
                self_intersection_extremes[0, 0] : self_intersection_extremes[0, 1],
                self_intersection_extremes[1, 0] : self_intersection_extremes[1, 1],
                self_intersection_extremes[2, 0] : self_intersection_extremes[2, 1],
            ] = operator(
                (grid_copy if default is not None else self.grid)[
                    self_intersection_extremes[0, 0] : self_intersection_extremes[0, 1],
                    self_intersection_extremes[1, 0] : self_intersection_extremes[1, 1],
                    self_intersection_extremes[2, 0] : self_intersection_extremes[2, 1],
                ],
                other.grid[
                    other_intersection_extremes[0, 0] : other_intersection_extremes[0, 1],
                    other_intersection_extremes[1, 0] : other_intersection_extremes[1, 1],
                    other_intersection_extremes[2, 0] : other_intersection_extremes[2, 1],
                ],
            )

        self.local_function(function, other)

    def intersection_(self, other: VoxelSolid) -> None:
        r"""Finds the intersection of two voxel solids and writes the result in the first one.

        Args:
            other (VoxelSolid):
                The other solid.
        """
        self.local_operator(np.logical_and, other, default=False)

    def intersection(self, other: VoxelSolid) -> VoxelSolid:
        r"""Finds the intersection of two voxel solids and returns it as a new solid.

        Args:
            other (VoxelSolid):
                The other solid.
        """
        new_voxel_solid: VoxelSolid = deepcopy(self)
        new_voxel_solid.intersection_(other)
        return new_voxel_solid

    def union_(self, other: VoxelSolid) -> None:
        r"""Finds the union of two voxel solids and writes the result in the first one.

        Args:
            other (VoxelSolid):
                The other solid.
        """
        self.local_operator(np.logical_or, other)

    def union(self, other: VoxelSolid) -> VoxelSolid:
        r"""Finds the union of two voxel solids and returns it as a new solid.

        Args:
            other (VoxelSolid):
                The other solid.
        """
        new_voxel_solid: VoxelSolid = deepcopy(self)
        new_voxel_solid.union_(other)
        return new_voxel_solid

    def voxel_volume(self) -> float:
        r"""Returns the volume of a voxel.

        Returns (float):
            The volume of a voxel.
        """
        return self.resolution**3

    def volume(self) -> float:
        r"""Returns the volume of the solid.

        Returns (float):
            The volume of the solid.
        """
        return self.int_volume() * self.voxel_volume()

    def int_volume(self) -> int:
        r"""Returns the number of voxels in the solid.

        It is the volume of the solid in the integer space.

        Returns (int):
            The number of voxels in the solid.
        """
        return np.count_nonzero(self.grid)
