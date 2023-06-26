from tqdm import tqdm
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable

from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt

from sadic.solid import Solid, Sphere, Multisphere
from sadic.pdb import PDBEntity
from sadic.quantizer import Quantizer, RegularStepsCartesianQuantizer

class VoxelSolid(Solid):
    default_quantizer_class = RegularStepsCartesianQuantizer
    default_quantizer_kwargs = {"steps_number": 32}

    def __init__(
            self,
            arg1: (Sequence[Sphere]
                   | PDBEntity
                   | PandasPdb
                   | Structure
                   | NDArray[np.float32]),
            arg2: None | NDArray[np.float32] = None,
            resolution: float = 0.3,
            extreme_coordinates: None | NDArray[np.float32] = None,
            align_with: "None | VoxelSolid" = None
            ) -> None:
       
        if isinstance(arg1, Multisphere):
            self.build_from_multisphere(
                arg1, resolution, extreme_coordinates=extreme_coordinates)
        else:
            multisphere = Multisphere(arg1, arg2)
            self.build_from_multisphere(
                multisphere, resolution,
                extreme_coordinates=extreme_coordinates, align_with=align_with)

    def build_from_multisphere(
            self,
            multisphere: Multisphere,
            resolution: float = 0.3,
            extreme_coordinates: None | NDArray[np.float32] = None,
            align_with: "None | VoxelSolid" = None
            ) -> None:
        
        self.multisphere = multisphere
        self.resolution = resolution

        if extreme_coordinates is None:
            self.extreme_coordinates = multisphere.get_extreme_coordinates()
            self.extreme_coordinates[:, 1] += self.resolution * np.modf(
                (self.extreme_coordinates[:, 1]
                 - self.extreme_coordinates[:, 0]) / self.resolution)[0]
            if align_with is not None:
                if self.resolution != align_with.resolution:
                    raise ValueError(
                        "resolution must be the same as align_with.resolution")
                grid_offset = (align_with.extreme_coordinates[:, 0]
                               - self.extreme_coordinates[:, 0]
                               ) / self.resolution
                self.extreme_coordinates = self.extreme_coordinates + (
                    (grid_offset - np.round(grid_offset))
                     * self.resolution).reshape(-1, 1)
        else:
            self.extreme_coordinates = extreme_coordinates

        self.dimensions = np.ceil(
            (self.extreme_coordinates[:, 1] - self.extreme_coordinates[:, 0])
            / self.resolution).astype(np.int32)
        self.grid = np.empty(self.dimensions, dtype=np.bool_)
            
        if len(self.multisphere) == 1:
            self.grid = multisphere.is_inside(self.grid_to_cartesian(
                self.get_all_grid_coordinates())).reshape(self.dimensions)
        else:
            for center, radius in tqdm(
                    zip(*self.multisphere.get_all_centers_and_radii())):
                sphere = VoxelSolid(
                    [Sphere(center, radius)],
                    resolution=self.resolution, align_with=self)
                self.union_(sphere)

    def get_all_grid_coordinates(self) -> NDArray[np.int32]:
        return np.mgrid[
            0:self.dimensions[0], 0:self.dimensions[1], 0:self.dimensions[2]
            ].astype(np.int32).transpose(1, 2, 3, 0).reshape(
            -1, self.dimensions.shape[0])

    def cartesian_to_grid(
            self,
            coordinates: NDArray[np.float32]) -> NDArray[np.int32]:
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D coordinates")
        
        return np.floor((coordinates - self.extreme_coordinates[:, 0])
                        / self.resolution).astype(np.int32)
    
    def grid_to_cartesian(
            self,
            coordinates: NDArray[np.int32]) -> NDArray[np.float32]:
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D indexes")
        
        return (coordinates * self.resolution + self.extreme_coordinates[:, 0]
                + self.resolution / 2)

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        return self.extreme_coordinates
    
    def remove_holes_from_grid(self) -> NDArray[np.bool_]:
        return label(self.grid.astype(np.int32))[0] != 0
    
    def remove_holes(self, *args, **kwargs) -> "VoxelSolid":
        vs = deepcopy(self)
        vs.grid = vs.remove_holes_from_grid(*args, **kwargs)
        return vs

    def edt(self) -> NDArray[np.float32]:
        return distance_transform_edt(self.grid, sampling=self.resolution)

    def translate(self, shift: NDArray[np.int32]) -> "VoxelSolid":
        if shift.ndim != 1 or shift.shape[0] != 3:
            raise ValueError("shift must be a 1D array of 3 elements")

        vs = deepcopy(self)
        
        vs.grid = np.roll(vs.grid, shift, axis=(0, 1, 2))

        for axis in range(3):
            array_slice = [slice(None)] * vs.grid.ndim
            if shift[axis] > 0:
                array_slice[axis] = slice(0, shift[axis])
                vs.grid[tuple(array_slice)] = False
            else:
                array_slice[axis] = slice(shift[axis], None)
                vs.grid[tuple(array_slice)] = False

        return vs

    def is_inside(
            self,
            arg: NDArray[np.float32] | Sphere,
            get_volumes: bool = False) -> NDArray[np.bool_]:
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg, get_volumes=get_volumes)
        
        elif isinstance(arg, np.ndarray):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a numpy.ndarray or a Sphere")

    def point_is_inside_boundaries(
            self,
            points: NDArray[np.float32]) -> NDArray[np.bool_]:
        return np.logical_and(
            np.logical_and(
                np.logical_and(points[:, 0] > self.extreme_coordinates[0, 0],
                               points[:, 0] < self.extreme_coordinates[0, 1]),
                np.logical_and(points[:, 1] > self.extreme_coordinates[1, 0],
                               points[:, 1] < self.extreme_coordinates[1, 1])
            ),
            np.logical_and(points[:, 2] > self.extreme_coordinates[2, 0],
                           points[:, 2] < self.extreme_coordinates[2, 1])
        )

    def point_is_inside(
            self,
            points: NDArray[np.float32]) -> NDArray[np.bool_]:
        output = self.point_is_inside_boundaries(points)
        grid_points = self.cartesian_to_grid(points[output])
        output[output] = self.grid[grid_points[:, 0], grid_points[:, 1],
                                   grid_points[:, 2]]
        return output

    def sphere_is_inside(
            self,
            sphere: Sphere,
            quantizer_arg: Quantizer | None = None,
            get_volumes: bool = False) -> NDArray[np.bool_]:
        quantizer: Quantizer

        if quantizer_arg is None:
            quantizer = Multisphere.default_quantizer_class(
                **Multisphere.default_quantizer_kwargs)
        else:
            quantizer = quantizer_arg

        points, _ = quantizer.get_points_and_volumes(sphere)

        if get_volumes:
            return self.point_is_inside(points)

        return self.point_is_inside(points)

    def local_function(self, function: Callable, other: "VoxelSolid") -> None:
         # check if grids are aligned
        if self.resolution != other.resolution:
            raise ValueError("Grids must have the same resolution")
        displacement = (self.extreme_coordinates - other.extreme_coordinates
                        ) / self.resolution
        if np.any(np.round(displacement- np.round(displacement, decimals=2),
                           decimals=2)):
            raise ValueError("Grids must be aligned")
        
        # find the intersection points of the grids
        min_intersection_centers = np.max(np.stack((
            self.extreme_coordinates[:, 0], other.extreme_coordinates[:, 0]),
            axis=1), axis=1) + self.resolution / 2
        max_intersection_centers = np.min(np.stack((
            self.extreme_coordinates[:, 1], other.extreme_coordinates[:, 1]),
            axis=1), axis=1) + self.resolution / 2

        self_intersection_centers = np.stack((
            self.cartesian_to_grid(min_intersection_centers),
            self.cartesian_to_grid(max_intersection_centers)), axis=1)
        other_intersection_centers = np.stack((
            other.cartesian_to_grid(min_intersection_centers),
            other.cartesian_to_grid(max_intersection_centers)), axis=1)

        function(self, other, self_intersection_centers,
                 other_intersection_centers)

    def local_operator(self, operator: Callable, other: "VoxelSolid") -> None:
        def function(
                self,
                other,
                self_intersection_extremes,
                other_intersection_extremes):
            self.grid[
                self_intersection_extremes[0, 0]
                :self_intersection_extremes[0, 1],
                self_intersection_extremes[1, 0]
                :self_intersection_extremes[1, 1],
                self_intersection_extremes[2, 0]
                :self_intersection_extremes[2, 1]
                ] = operator(
                self.grid[
                    self_intersection_extremes[0, 0]
                    :self_intersection_extremes[0, 1],
                    self_intersection_extremes[1, 0]
                    :self_intersection_extremes[1, 1],
                    self_intersection_extremes[2, 0]
                    :self_intersection_extremes[2, 1]
                ],
                other.grid[
                    other_intersection_extremes[0, 0]
                    :other_intersection_extremes[0, 1],
                    other_intersection_extremes[1, 0]
                    :other_intersection_extremes[1, 1],
                    other_intersection_extremes[2, 0]
                    :other_intersection_extremes[2, 1]
                ])
            
        self.local_function(function, other)

    def intersection_(self, other: "VoxelSolid") -> None:
        self.local_operator(np.logical_and, other)

    def intersection(self, other: "VoxelSolid") -> "VoxelSolid":
        vs = deepcopy(self)
        vs.intersection_(other)
        return vs

    def union_(self, other: "VoxelSolid") -> None:
        self.local_operator(np.logical_or, other)

    def union(self, other: "VoxelSolid") -> "VoxelSolid":
        vs = deepcopy(self)
        vs.union_(other)
        return vs
        
    def voxel_volume(self) -> float:
        return self.resolution ** 3
    
    def volume(self) -> float:
        return self.int_volume() * self.voxel_volume()
    
    def int_volume(self) -> int:
        return np.count_nonzero(self.grid)