from Solid import Solid
from Sphere import Sphere
from PDBEntity import PDBEntity
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from collections.abc import Sequence, Iterable
from numbers import Number

import numpy as np
from numpy.typing import NDArray

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
            if isinstance(arg1, Sequence) and isinstance(arg2, Sequence):
                if isinstance(arg1[0], Sequence) and isinstance(arg1[0][0], Number) and isinstance(arg2[0], Number):
                    self.create_from_centers_and_radii(arg1, arg2)
                
                else:
                    raise TypeError("2 arguments must be sequences of points and radii")
            
            else:
                raise TypeError("2 arguments must be sequences of points and radii")

        else:
            raise TypeError("Expected 1 or 2 arguments, got {}".format(len(args)))

    def create_empty(self, length: int):
        self.centers: NDArray[np.float32] = np.empty((length, 3), dtype=np.float32)
        self.radii: NDArray[np.float32] = np.empty((length,), dtype=np.float32)
        self.voronoi = None

    def create_from_centers_and_radii(self, centers: Sequence[Sequence[Number]], radii: Sequence[Number]):
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

    def is_inside(self, arg) -> NDArray[np.bool_]:
        element = arg[0] if isinstance(arg, Sequence) else arg
        
        if isinstance(element, Sequence):
            return self.point_is_inside(element)

        elif isinstance(element, Sphere):
            return self.sphere_is_inside(element)

        else:
            raise TypeError("Argument must be a sequence or a Sphere")

    def point_is_inside(self, points: Sequence[Number] | Sequence[Sequence[Number]]) -> NDArray[np.bool_]:
        raise NotImplementedError()

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

    def get_points(self):
        raise NotImplementedError()

    def get_surface_points(self):
        print("Warning: get_surface_points is not implemented, it currently returns all points")
        return self.get_points()

    def get_internal_points(self):
        print("Warning: get_internal_points is not implemented, it currently returns all points")
        return self.get_points()