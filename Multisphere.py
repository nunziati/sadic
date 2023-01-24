from Solid import Solid
from Sphere import Sphere
from PDBEntity import PDBEntity
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from .typing import PointType, PointSequenceType, NumberSequenceType
from .typing import is_PointType, is_PointSequenceType, is_NumberSequenceType

from utils import point_square_distance

from collections.abc import Sequence
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Multisphere(Solid):
    def __init__(self, *args):
        super().__init__(*args)

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Sequence):
                if isinstance(arg[0], Sphere):
                    self.create_from_spheres(arg)
                else:
                    raise TypeError("Sequence argument must be a sequence of Spheres")
            
            elif isinstance(arg, PDBEntity):
                self.create_from_sadic_protein(arg)

            elif isinstance(arg, PandasPdb):
                self.create_from_biopandas_protein(arg)

            elif isinstance(arg, Structure):
                self.create_from_biopython_protein(arg)

            else:
                raise TypeError("Single argument must be a sequence of Spheres, PDBEntity, PandasPdb or Structure")

        elif len(args) == 2:
            arg1, arg2 = args
            if is_PointSequenceType(arg1) and is_NumberSequenceType(arg2):
                self.create_from_centers_and_radii(arg1, arg2)
            
            else:
                raise TypeError("2 arguments must be sequences of points and radii")

        else:
            raise TypeError("Expected 1 or 2 arguments, got {}".format(len(args)))

    def create_empty(self, length: int):
        self.centers: NDArray[np.float32] = np.empty((length, 3), dtype=np.float32)
        self.radii: NDArray[np.float32] = np.empty((length,), dtype=np.float32)
        self.voronoi = None

    def create_from_centers_and_radii(self, centers: PointSequenceType, radii: NumberSequenceType):
        if len(centers) == 0 or len(radii) == 0:
            raise ValueError("points and radii must be non-empty")
        if len(centers) != len(radii):
            raise ValueError("points and radii must have the same length")
        for point in centers:
            if len(point) != 3:
                raise ValueError("Each point in points must be a sequence of length 3")

        self.create_empty(len(centers))

        for idx, (point, radius) in enumerate(zip(centers, radii)):
            self.centers[idx] = point
            self.radii[idx] = radius

    def create_from_spheres(self, spheres: Sequence[Sphere]):
        if len(spheres) == 0:
            raise ValueError("spheres must be non-empty")

        self.create_empty(len(spheres))

        for idx, sphere in enumerate(spheres):
            self.centers[idx] = sphere.center
            self.radii[idx] = sphere.radius

    def create_from_sadic_protein(self, protein: PDBEntity):
        self.centers: NDArray[np.float32] = protein.get_centers()
        self.radii: NDArray[np.float32] = protein.get_radii()

    def create_from_biopython_protein(self, protein: Structure):
        raise NotImplementedError()

    def create_from_biopandas_protein(self, protein: PandasPdb):
        sadic_protein = PDBEntity(protein)
        self.create_from_sadic_protein(sadic_protein)

    def is_inside(self, arg) -> bool | NDArray[np.bool_]:
        element = arg[0] if isinstance(arg, Sequence) else arg

        if isinstance(element, Sequence):
            return self.point_is_inside(element)

        elif isinstance(element, Sphere):
            return self.sphere_is_inside(element)

        else:
            raise TypeError("Argument must be a sequence or a Sphere")

    def single_point_is_inside(self, input_point: PointType) -> bool:
        if not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3")
        
        candidate = self.get_candidate_centers_and_radii(input_point, subset="best")        
        """
        candidate può avere diverse cose dentro:
        - un solo centro e relativo raggio
        - uno zip che itera su (centri, raggi)
        """
        for point, radius in candidate:
            if point_square_distance(input_point, point) <= radius ** 2:
                return True

        return False

    def point_is_inside(self, points) -> bool | NDArray[np.bool_]:
        if is_PointType(points[0]):
            return self.single_point_is_inside(points)

        if len(points) == 0:
            raise ValueError("points must be non-empty")
        
        for point in points:
            if is_PointType(point):
                raise ValueError("Each point in points must be a sequence of length 3")

        output: NDArray[np.bool_] = np.empty(len(points), dtype=np.bool_)
        for idx, input in enumerate(points):
            output[idx] = self.single_point_is_inside(input)

        return output

    def sphere_is_inside(self, sphere: Sphere | Sequence[Sphere]) -> NDArray[np.bool_]:
        raise NotImplementedError()

    def is_buried(self, arg):
        raise NotImplementedError()

    def point_is_buried(self, points):
        raise NotImplementedError()

    def sphere_is_buried(self, sphere):
        raise NotImplementedError()

    def compute_voronoi(self):
        raise NotImplementedError()

    def get_all_centers(self):
        return self.centers
    
    def get_all_radii(self):
        return self.radii

    def get_all_centers_and_radii(self):
        return zip(self.get_all_centers(), self.get_all_radii())

    def get_voronoi_center(self, input_point: PointType):
        raise NotImplementedError()

    def get_voronoi_radius(self, arg: int | PointType):
        raise NotImplementedError()
    
    def get_voronoi_center_and_radius(self, input_point: PointType | None):
        if input_point is None:
            raise Exception("input_point cannot be None")

        if self.voronoi is None:
            raise Exception("Must call compute_voronoi before calling get_voronoi_center_and_radius")
        
        if not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3 or None")

        return self.get_voronoi_center(input_point), self.get_voronoi_radius(input_point)

    def get_surface_centers(self):
        print("Warning: get_surface_centers is not implemented, it currently returns all points")
        return self.get_all_centers()

    def get_surface_radii(self):
        print("Warning: get_surface_radii is not implemented, it currently returns all radii")
        return self.get_all_radii()

    def get_surface_centers_and_radii(self):
        return zip(self.get_surface_centers(), self.get_surface_radii())

    def get_internal_centers(self):
        print("Warning: get_internal_centers is not implemented, it currently returns all points")
        return self.get_all_centers()

    def get_internal_radii(self):
        print("Warning: get_internal_radii is not implemented, it currently returns all radii")
        return self.get_all_radii()
        
    def get_internal_centers_and_radii(self):
        return zip(self.get_internal_centers(), self.get_internal_radii())

    def get_candidate_centers_and_radii(self, input_point: PointType | None = None, subset="best"):
        if input_point and not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3 or None")

        if subset == "all":
            return self.get_all_centers_and_radii()
        elif subset == "voronoi":
            return self.get_voronoi_center_and_radius(input_point)
        elif subset == "surface":
            return self.get_surface_centers_and_radii()
        elif subset == "internal":
            return self.get_internal_centers_and_radii()
        elif subset == "best":
            if input_point is None or self.voronoi is None or not is_PointType(input_point):
                return self.get_candidate_centers_and_radii(subset="all")
            else:
                return self.get_candidate_centers_and_radii(input_point, subset="voronoi")
        else:
            raise ValueError("subset must be one of 'all', 'voronoi', 'surface', 'internal', 'best'")