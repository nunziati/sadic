from collections.abc import Sequence
from typing import Type

from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from sadic.solid import Solid, Sphere
from sadic.pdb import PDBEntity
from sadic.quantizer import Quantizer, RegularStepsCartesianQuantizer

class Multisphere(Solid):
    default_quantizer_class: Type[Quantizer] = RegularStepsCartesianQuantizer
    default_quantizer_kwargs: dict[str, int] = {"steps_number": 32}

    def __init__(
            self,
            arg1: (Sequence[Sphere]
                   | PDBEntity
                   | PandasPdb
                   | Structure
                   | NDArray[np.float32]),
            arg2: None | NDArray[np.float32] = None ) -> None:
        if arg2 is None:
            if isinstance(arg1, Sequence):
                if isinstance(arg1[0], Sphere):
                    self.build_from_spheres(arg1)
                else:
                    raise TypeError(
                        "Sequence argument must be a sequence of Spheres")
            elif isinstance(arg1, PDBEntity):
                self.build_from_sadic_protein(arg1)
            elif isinstance(arg1, PandasPdb):
                self.build_from_biopandas_protein(arg1)
            elif isinstance(arg1, Structure):
                self.build_from_biopython_protein(arg1)
            else:
                raise TypeError(
                    "Single argument must be a sequence of Spheres, "
                    "PDBEntity, PandasPdb or Structure")
        else:
            if isinstance(arg1, np.ndarray):
                self.build_from_centers_and_radii(arg1, arg2)
            
            else:
                raise TypeError("2 arguments must be numpy.ndarray objects")        

    def build_empty(self, length: int) -> None:
        # NOTA: NON C'è BISOGNO DI COSTRUIRLA VUOTA TUTTE LE VOLTE, PERò BISOGNA COMUNQUE SETTARE A None I DUE ATTRIBUTI QUI SOTTO
        if length <= 0:
            raise ValueError("length must be a positive integer")

        self.centers: NDArray[np.float32] = np.empty((length, 3),
                                                     dtype=np.float32)
        self.radii: NDArray[np.float32] = np.empty((length,),
                                                   dtype=np.float32)
        self.extreme_coordinates: NDArray[np.float32] | None = None

    def build_from_centers_and_radii(
            self,
            centers: NDArray[np.float32],
            radii: NDArray[np.float32]) -> None:
        if centers.shape[1] != 3:
            raise ValueError(
                "centers must be numpy.ndarray objects with shape (n, 3)")

        if centers.shape[0] != radii.shape[0]:
            raise ValueError(
                "first argument and second argument must have the same number "
                "of rows")

        self.build_empty(centers.shape[0])

        self.centers: NDArray[np.float32] = centers
        self.radii: NDArray[np.float32] = radii

    def build_from_spheres(self, spheres: Sequence[Sphere]) -> None:
        if len(spheres) == 0:
            raise ValueError("spheres must be non-empty")

        self.build_empty(len(spheres))

        idx: int
        sphere: Sphere
        for idx, sphere in enumerate(spheres):
            self.centers[idx] = sphere.center
            self.radii[idx] = sphere.radius

    def build_from_sadic_protein(self, protein: PDBEntity) -> None:
        self.build_empty(len(protein))
        self.centers: NDArray[np.float32] = protein.get_centers()
        self.radii: NDArray[np.float32] = protein.get_radii()

    def build_from_biopython_protein(self, protein: Structure) -> None:
        sadic_protein: PDBEntity = PDBEntity(protein)
        self.build_from_sadic_protein(sadic_protein)

    def build_from_biopandas_protein(self, protein: PandasPdb) -> None:
        sadic_protein: PDBEntity = PDBEntity(protein)
        self.build_from_sadic_protein(sadic_protein)

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        ndim: int = 3

        extreme_coordinates: NDArray[np.float32] = np.empty((ndim, 2),
                                                            dtype=np.float32)
        axis: int
        for axis in range(ndim):
            extreme_coordinates[axis, 0] = np.min(
                self.centers[:, axis] - self.radii)
            extreme_coordinates[axis, 1] = np.max(
                self.centers[:, axis] + self.radii)

        return extreme_coordinates

    def is_inside(
            self,
            arg: NDArray[np.float32] | Sphere,
            get_volumes: bool = False) -> NDArray[np.bool_]:
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg, get_volumes=get_volumes)
        elif isinstance(arg, np.ndarray):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a numpy.ndarray or a Sphere")

    def point_is_inside(
            self,
            points: NDArray[np.float32]) -> NDArray[np.bool_]:
        if points.shape[0] <= 0:
            raise ValueError("points must be non-empty")
        
        if points.shape[1] != 3:
            raise ValueError("points must be a numpy.ndarray with shape (n, 3)")

        return (cdist(points.reshape((-1, 3)),
                      self.centers.reshape((-1, 3)),
                      metric='sqeuclidean') <= self.radii ** 2).any(axis=1)

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

        points: NDArray[np.float32]
        points, _ = quantizer.get_points_and_volumes(sphere)

        if get_volumes:
            return self.point_is_inside(points)

        return self.point_is_inside(points)

    def get_all_centers(self):
        return self.centers
    
    def get_all_radii(self):
        return self.radii

    def get_all_centers_and_radii(self):
        return self.get_all_centers(), self.get_all_radii()
        
    def __len__(self) -> int:
        return self.centers.shape[0]