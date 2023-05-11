from sadic.solid import Solid, Sphere, Multisphere
from sadic.pdb import PDBEntity
from sadic.quantizer import Quantizer, RegularStepsCartesianQuantizer

from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from tqdm import tqdm

from collections.abc import Sequence

import numpy as np

from numpy.typing import NDArray

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt

from copy import deepcopy

class VoxelSolid(Solid):
    default_quantizer_class = RegularStepsCartesianQuantizer
    default_quantizer_kwargs = {"steps_number": 32}

    def __init__(self, arg1: Sequence[Sphere] | PDBEntity | PandasPdb | Structure | NDArray[np.float32], arg2: None | NDArray[np.float32] = None, resolution: float = 0.3) -> None:
       multisphere = Multisphere(arg1, arg2)
       self.build_from_multisphere(multisphere, resolution)

    def build_from_multisphere(self, multisphere: Multisphere, resolution: float = 0.3) -> None:
        self.multisphere = multisphere
        self.resolution = resolution
        self.extreme_coordinates = multisphere.get_extreme_coordinates()
        self.dimensions = np.ceil((self.extreme_coordinates[:, 1] - self.extreme_coordinates[:, 0]) / resolution).astype(np.int32)
        self.grid = np.empty(self.dimensions, dtype=np.bool_)

        yz_coordinates = np.array(np.meshgrid(np.arange(self.dimensions[1]), np.arange(self.dimensions[2]))).T.reshape(-1, 2)

        for x in tqdm(range(self.dimensions[0])):
            points = np.concatenate([np.full((self.dimensions[1] * self.dimensions[2], 1), x), yz_coordinates], axis=-1).astype(np.int32)
            self.grid[points[:, 0], points[:, 1], points[:, 2]] = multisphere.is_inside(self.grid_to_cartesian(points), get_volumes=False)

    def cartesian_to_grid(self, coordinates: NDArray[np.float32]) -> NDArray[np.int32]:
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D coordinates")
        
        return np.floor((coordinates - self.extreme_coordinates[:, 0]) / self.resolution).astype(np.int32)
    
    def grid_to_cartesian(self, coordinates: NDArray[np.int32]) -> NDArray[np.float32]:
        if coordinates.shape[coordinates.ndim - 1] != 3:
            raise ValueError("coordinates must be an array of 3D indexes")
        
        return coordinates * self.resolution + self.extreme_coordinates[:, 0] + self.resolution / 2

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

    def is_inside(self, arg: NDArray[np.float32] | Sphere, get_volumes: bool = False) -> NDArray[np.bool_]:
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg, get_volumes=get_volumes)
        
        elif isinstance(arg, np.ndarray):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a numpy.ndarray or a Sphere")

    def point_is_inside_boundaries(self, points: NDArray[np.float32]) -> NDArray[np.bool_]:
        return np.logical_and(
            np.logical_and(
                np.logical_and(points[:, 0] > self.extreme_coordinates[0, 0], points[:, 0] < self.extreme_coordinates[0, 1]),
                np.logical_and(points[:, 1] > self.extreme_coordinates[1, 0], points[:, 1] < self.extreme_coordinates[1, 1])
            ),
            np.logical_and(points[:, 2] > self.extreme_coordinates[2, 0], points[:, 2] < self.extreme_coordinates[2, 1])
        )

    def point_is_inside(self, points: NDArray[np.float32]) -> NDArray[np.bool_]:
        output = self.point_is_inside_boundaries(points)
        grid_points = self.cartesian_to_grid(points[output])
        output[output] = self.grid[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]]
        return output

    def sphere_is_inside(self, sphere: Sphere, quantizer_arg: Quantizer | None = None, get_volumes: bool = False) -> NDArray[np.bool_]:
        quantizer: Quantizer = Multisphere.default_quantizer_class(**Multisphere.default_quantizer_kwargs) if quantizer_arg is None else quantizer_arg

        points, _ = quantizer.get_points_and_volumes(sphere)

        if get_volumes:
            return self.point_is_inside(points)

        return self.point_is_inside(points)

    def get_surface_centers(self):
        raise NotImplementedError

    def get_surface_radii(self):
        raise NotImplementedError

    def get_surface_centers_and_radii(self):
        return self.get_surface_centers(), self.get_surface_radii()

    def get_internal_centers(self):
        raise NotImplementedError

    def get_internal_radii(self):
        raise NotImplementedError
        
    def get_internal_centers_and_radii(self):
        return self.get_internal_centers(), self.get_internal_radii()

    def get_candidate_centers_and_radii(self, input_point = None, subset = "best"):
        """if input_point and not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3 or None")
        
        if subset == "best":
            if input_point is None or self.voronoi is None or not is_PointType(input_point):
                subset = "all"
            else:
                subset = "voronoi"

        if subset == "all":
            return self.get_all_centers_and_radii()
        elif subset == "voronoi":
            if input_point is None:
                raise Exception("Input point must not be None")
            voronoi_center: PointSequenceType
            voronoi_radius: NumberSequenceType
            _, voronoi_center, voronoi_radius = self.get_voronoi_center_and_radius(input_point)
            return voronoi_center, voronoi_radius
        elif subset == "surface":
            return self.get_surface_centers_and_radii()
        elif subset == "internal":
            return self.get_internal_centers_and_radii()
        else:
            raise ValueError("subset must be one of 'all', 'voronoi', 'surface', 'internal', 'best'")"""