from Solid import Solid
from Sphere import Sphere
from PDBEntity import PDBEntity
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from tqdm import tqdm

from utils import point_square_distance

from collections.abc import Sequence

import numpy as np

from numpy.typing import NDArray
import typing as tp

# RINOMINA I FILE IN MINUSCOLO
# IMPORTA DIRETTAMENTE I MODULI INTERI, INVECE DELLE FUNZIONI/CLASSI
# METTI TUTTI I SOLIDI NELLO STESSO FILE

class Multisphere(Solid):
    def __init__(self, arg1: Sequence[Sphere] | PDBEntity | PandasPdb | Structure | NDArray[np.float32], arg2: None | NDArray[np.float32] = None ) -> None:
        if arg2 is None:
            if isinstance(arg1, Sequence):
                if isinstance(arg1[0], Sphere):
                    self.build_from_spheres(arg1)
                else:
                    raise TypeError("Sequence argument must be a sequence of Spheres")
            
            elif isinstance(arg1, PDBEntity):
                self.build_from_sadic_protein(arg1)

            elif isinstance(arg1, PandasPdb):
                self.build_from_biopandas_protein(arg1)

            elif isinstance(arg1, Structure):
                self.build_from_biopython_protein(arg1)

            else:
                raise TypeError("Single argument must be a sequence of Spheres, PDBEntity, PandasPdb or Structure")

        else:
            if isinstance(arg1, np.ndarray):
                self.build_from_centers_and_radii(arg1, arg2)
            
            else:
                raise TypeError("2 arguments must be numpy.ndarray objects")        

    def build_empty(self, length: int) -> None:
        self.centers: NDArray[np.float32] = np.empty((length, 3), dtype=np.float32)
        self.radii: NDArray[np.float32] = np.empty((length,), dtype=np.float32)
        self.voronoi: None = None
        self.extreme_coordinates: NDArray[np.float32] | None = None

    def build_from_centers_and_radii(self, centers: NDArray[np.float32], radii: NDArray[np.float32]) -> None:
        if centers.shape[1] != 3:
            raise ValueError("centers must be numpy.ndarray objects with shape (n, 3)")

        if radii.shape[1]!= 3:
            raise ValueError("radii must be numpy.ndarray objects with shape (n,)")

        if centers.shape[0] != radii.shape[0]:
            raise ValueError("first argument and second argument must have the same number of rows")

        self.centers: NDArray[np.float32] = centers
        self.radii: NDArray[np.float32] = radii

    def build_from_spheres(self, spheres: Sequence[Sphere]) -> None:
        if len(spheres) == 0:
            raise ValueError("spheres must be non-empty")

        self.build_empty(len(spheres))

        for idx, sphere in enumerate(spheres):
            self.centers[idx] = sphere.center
            self.radii[idx] = sphere.radius

    def build_from_sadic_protein(self, protein: PDBEntity) -> None:
        self.centers: NDArray[np.float32] = protein.get_centers()
        self.radii: NDArray[np.float32] = protein.get_radii()

    def build_from_biopython_protein(self, protein: Structure) -> None:
        sadic_protein: PDBEntity = PDBEntity(protein)
        self.build_from_sadic_protein(sadic_protein)

    def build_from_biopandas_protein(self, protein: PandasPdb) -> None:
        sadic_protein: PDBEntity = PDBEntity(protein)
        self.build_from_sadic_protein(sadic_protein)

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        pass

    def is_inside_fast(self, points):
        return (((points.reshape((-1, 1, 3)) - self.centers.reshape((1, -1, 3))) ** 2).sum(axis=-1) <= self.radii.reshape((1, -1)) ** 2).any(axis=1)
    
    def is_inside_exclusive(self, points):
        print(points.shape)
        output = np.empty(points.shape, dtype=bool)

        for idx, point in tqdm(enumerate(points)):
            for center, radius in zip(self.centers, self.radii):
                if ((point - center) ** 2).sum() < radius ** 2:
                    output[idx] = True
                    break

        return output

    def is_inside(self, arg: PointSequenceType | Sphere) -> NDArray[np.bool_]:
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg)

        if is_PointSequenceType(arg):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a sequence or a Sphere")

    def point_is_inside(self, points: PointSequenceType) -> NDArray[np.bool_]:
        if len(points) == 0:
            raise ValueError("points must be non-empty")
        
        for point in points:
            if is_PointType(point):
                raise ValueError("Each point in points must be a sequence of length 3")

        output: NDArray[np.bool_] = np.empty(len(points), dtype=np.bool_)
        for idx, point in enumerate(points):
            output[idx] = self.single_point_is_inside(point)

        return output

    def single_point_is_inside(self, input_point: PointType) -> bool:
        if not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3")
        
        candidate_centers: PointSequenceType
        candidate_radii: NumberSequenceType
        candidate_centers, candidate_radii = self.get_candidate_centers_and_radii(input_point, subset="best")        

        for point, radius in zip(candidate_centers, candidate_radii):
            if point_square_distance(input_point, point) <= radius ** 2:
                return True

        return False

    def sphere_is_inside(self, sphere: Sphere) -> NDArray[np.bool_]:
        raise NotImplementedError

    """def is_buried(self, arg) -> bool:
        raise NotImplementedError

    def point_is_buried(self, points) -> bool:
        raise NotImplementedError

    def sphere_is_buried(self, sphere) -> bool:
        raise NotImplementedError"""

    def compute_voronoi(self) -> None:
        raise NotImplementedError

    def get_all_centers(self):
        return self.centers
    
    def get_all_radii(self):
        return self.radii

    def get_all_centers_and_radii(self):
        return self.get_all_centers(), self.get_all_radii()

    def get_voronoi_center(self, input_point: PointType) -> tuple[int, PointSequenceType]:
        raise NotImplementedError

    def get_voronoi_radius(self, arg: int | PointType) -> NumberSequenceType:
        raise NotImplementedError
    
    def get_voronoi_center_and_radius(self, input_point: PointType) -> tuple[int, PointSequenceType, NumberSequenceType]:
        if self.voronoi is None:
            raise Exception("Must call compute_voronoi before calling get_voronoi_center_and_radius")
        
        if not is_PointType(input_point):
            raise TypeError("Argument must be a sequence of length 3 or None")

        voronoi_index: int
        voronoi_center: PointSequenceType
        voronoi_index, voronoi_center = self.get_voronoi_center(input_point)
        voronoi_radius: NumberSequenceType = self.get_voronoi_radius(voronoi_index)
        
        return voronoi_index, voronoi_center, voronoi_radius

    def get_surface_centers(self) -> PointSequenceType:
        raise NotImplementedError

    def get_surface_radii(self) -> NumberSequenceType:
        raise NotImplementedError

    def get_surface_centers_and_radii(self) -> tuple[PointSequenceType, NumberSequenceType]:
        return self.get_surface_centers(), self.get_surface_radii()

    def get_internal_centers(self) -> PointSequenceType:
        raise NotImplementedError

    def get_internal_radii(self) -> NumberSequenceType:
        raise NotImplementedError
        
    def get_internal_centers_and_radii(self) -> tuple[PointSequenceType, NumberSequenceType]:
        return self.get_internal_centers(), self.get_internal_radii()

    def get_candidate_centers_and_radii(self, input_point: PointType | None = None, subset: str = "best") -> tuple[PointSequenceType, NumberSequenceType]:
        if input_point and not is_PointType(input_point):
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
            raise ValueError("subset must be one of 'all', 'voronoi', 'surface', 'internal', 'best'")