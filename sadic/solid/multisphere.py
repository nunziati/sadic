r"""Defines the Multisphere class."""

from collections.abc import Sequence
from typing import Type

from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from sadic.solid import Solid, Sphere
from sadic.pdb import Model
from sadic.quantizer import Quantizer, RegularStepsCartesianQuantizer


class Multisphere(Solid):
    r"""Solid representing the union of many spheres.

    Can be used to check if a (set of) point(s) is inside the set of spheres. Can also be used to
    get the extreme coordinates of the set of spheres. The spheres can be built from a set of
    spheres, a sadic.Model, a biopandas Structure representing a protein or from a set of centers
    and radii.

    Attributes:
        centers (NDArray[np.float32]):
            A numpy array of floats representing the centers of the spheres.
        radii (NDArray[np.float32]):
            A numpy array of floats representing the radii of the spheres.
        extreme_coordinates (NDArray[np.float32]):
            A numpy array of floats representing the extreme coordinates of the set of spheres.

    Methods:
        __init__:
            Constructor for multispheres.
        build_empty:
            Method to build an empty multisphere.
        build_from_spheres:
            Method to build the multisphere from a set of spheres.
        build_from_sadic_protein:
            Method to build the multisphere from a sadic.Model.
        build_from_structure:
            Method to build the multisphere from a biopandas Structure.
        build_from_centers_and_radii:
            Method to build the multisphere from a set of centers and radii.
        is_inside:
            Method to check if a (set of) point(s) or a sphere is inside the multisphere.
        point_is_inside:
            Method to check if a (set of) point(s) is inside the multisphere.
        sphere_is_inside:
            Method to check if a sphere is inside the multisphere.
        get_extreme_coordinates:
            Method to get the extreme coordinates of the multisphere.
        get_all_centers:
            Method to get all the centers of the spheres composing the multisphere.
        get_all_radii:
            Method to get all the radii of the spheres composing the multisphere.
        get_all_centers_and_radii:
            Method to get all the centers and radii of the spheres composing the multisphere.
        __len__:
            Method to get the number of spheres composing the multisphere.

    """
    default_quantizer_class: Type[Quantizer] = RegularStepsCartesianQuantizer
    default_quantizer_kwargs: dict[str, int] = {"steps_number": 32}

    def __init__(
        self,
        arg1: (Sequence[Sphere] | Model | Structure | NDArray[np.float32]),
        arg2: None | NDArray[np.float32] = None,
    ) -> None:
        r"""Initializes a multisphere based on the given argument(s).

        The multisphere can be built from a set of spheres, a Model, a biopandas Structure
        representing a protein or from a set of centers and radii. The build method is chosen based
        on the type of the first argument.

        Args:
            arg1 (Sequence[Sphere] | Model | Structure | NDArray[np.float32]):
                The first argument can be a sequence of spheres, a Model, a biopandas Structure
                representing a protein or a numpy array of floats representing the centers of the
                spheres.
            arg2 (None | NDArray[np.float32]):
                The second argument can be a numpy array of floats representing the radii of the
                spheres. It is only used if the first argument is a numpy array of floats
                representing the centers of the spheres.

        Raises:
            TypeError:
                If the types of the arguments are not valid for building the multisphere using any
                of the possible methods.
        """
        self.centers: NDArray[np.float32]
        self.radii: NDArray[np.float32]
        self.extreme_coordinates: NDArray[np.float32] | None

        if arg2 is None:
            if isinstance(arg1, Sequence):
                if isinstance(arg1[0], Sphere):
                    self.build_from_spheres(arg1)
                else:
                    raise TypeError("Sequence argument must be a sequence of Spheres")
            elif isinstance(arg1, Model):
                self.build_from_sadic_protein(arg1)
            elif isinstance(arg1, Structure):
                self.build_from_biopython_protein(arg1)
            else:
                raise TypeError("Single argument must be a sequence of Spheres, Model or Structure")
        else:
            if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
                self.build_from_centers_and_radii(arg1, arg2)

            raise TypeError("2 arguments must be numpy.ndarray objects")

    def build_empty(self, length: int) -> None:
        r"""Builds an empty multisphere.

        An empty multisphere is a multisphere with no spheres. The multisphere is initialized with
        the given length: the arrays representing the centers and radii of the spheres are
        initialized with the given length.

        Args:
            length (int):
                The length of the multisphere, i.e. the number of spheres composing the multisphere.

        Raises:
            ValueError:
                If the length is not a positive integer.
        """
        if length <= 0:
            raise ValueError("length must be a positive integer")

        self.centers = np.empty((length, 3), dtype=np.float32)
        self.radii = np.empty((length,), dtype=np.float32)
        self.extreme_coordinates = None

    def build_from_centers_and_radii(
        self, centers: NDArray[np.float32], radii: NDArray[np.float32]
    ) -> None:
        r"""Builds the multisphere from a set of centers and radii.

        Args:
            centers (NDArray[np.float32]):
                The centers of the spheres composing the multisphere.
            radii (NDArray[np.float32]):
                The radii of the spheres composing the multisphere.

        Raises:
            ValueError:
                If the arrays representing the centers and radii are not valid i.e. if they are not
                numpy.ndarray objects, if they are not of type np.float32, if they do not have the
                same number of rows or if the centers are not 3-dimensional.
        """
        if centers.shape[1] != 3:
            raise ValueError("centers must be numpy.ndarray objects with shape (n, 3)")

        if centers.shape[0] != radii.shape[0]:
            raise ValueError("first argument and second argument must have the same number of rows")

        self.build_empty(centers.shape[0])

        self.centers: NDArray[np.float32] = centers
        self.radii: NDArray[np.float32] = radii

    def build_from_spheres(self, spheres: Sequence[Sphere]) -> None:
        r"""Builds the multisphere from a sequence of spheres.

        Args:
            spheres (Sequence[Sphere]):
                The spheres composing the multisphere.

        Raises:
            ValueError:
                If the sequence of spheres is empty.
        """
        if len(spheres) == 0:
            raise ValueError("spheres must be non-empty")

        self.build_empty(len(spheres))

        idx: int
        sphere: Sphere
        for idx, sphere in enumerate(spheres):
            self.centers[idx] = sphere.center
            self.radii[idx] = sphere.radius

    def build_from_sadic_protein(self, protein: Model) -> None:
        r"""Builds the multisphere from a Model.

        Args:
            protein (Model):
                The Model representing the protein.
        """
        self.build_empty(len(protein))
        self.centers: NDArray[np.float32] = protein.get_centers()
        self.radii: NDArray[np.float32] = protein.get_radii()

    def build_from_biopython_protein(self, protein: Structure) -> None:
        r"""Builds the multisphere from a biopython Structure.

        To be implemented.
        """
        raise NotImplementedError

    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        r"""Returns the extreme coordinates of the multisphere.

        The extreme coordinates of a multisphere are the coordinates of the corners of the smallest
        axis-aligned bounding box containing the multisphere. They are computed by taking the
        minimum and maximum coordinates of the centers of the spheres composing the multisphere and
        subtracting and adding the corresponding radii.

        Returns:
            NDArray[np.float32]:
                A numpy.ndarray object of shape (3, 2) containing the extreme coordinates of the
                multisphere.
        """
        ndim: int = 3

        extreme_coordinates: NDArray[np.float32] = np.empty((ndim, 2), dtype=np.float32)
        axis: int
        for axis in range(ndim):
            extreme_coordinates[axis, 0] = np.min(self.centers[:, axis] - self.radii)
            extreme_coordinates[axis, 1] = np.max(self.centers[:, axis] + self.radii)

        return extreme_coordinates

    def is_inside(self, *args, **kwargs) -> NDArray[np.bool_]:
        r"""Checks if a (set of) point(s) or a sphere is inside the multisphere.

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
            of the sphere.
        """
        # TO DO: implement the get_volume option
        arg: NDArray[np.float32] | Sphere = args[0]
        get_volumes: bool = kwargs.get("get_volumes", False)
        if isinstance(arg, Sphere):
            return self.sphere_is_inside(arg, get_volumes=get_volumes)
        elif isinstance(arg, np.ndarray):
            return self.point_is_inside(arg)

        raise TypeError("Argument must be a numpy.ndarray or a Sphere")

    def point_is_inside(self, points: NDArray[np.float32]) -> NDArray[np.bool_]:
        r"""Checks if a set of points is inside the multisphere.

        Args:
            points (NDArray[np.float32]):
                The points to check.

        Returns (NDArray[np.bool_]):
            A numpy.ndarray object of shape (n,) containing the result of the check for each point.
        """
        if points.shape[0] <= 0:
            raise ValueError("points must be non-empty")

        if points.shape[1] != 3:
            raise ValueError("points must be a numpy.ndarray with shape (n, 3)")

        return (
            cdist(points.reshape((-1, 3)), self.centers.reshape((-1, 3)), metric="sqeuclidean")
            <= self.radii**2
        ).any(axis=1)

    def sphere_is_inside(
        self, sphere: Sphere, quantizer_arg: Quantizer | None = None, get_volumes: bool = False
    ) -> NDArray[np.bool_]:
        r"""Checks if a sphere is inside the multisphere.

        Quantizes the sphere and checks if the quantized points are inside the multisphere. The
        method can also return the volumes of the quantized cells containing the points of the
        sphere.

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

    def get_all_centers(self) -> NDArray[np.float32]:
        r"""Returns the centers of the spheres composing the multisphere.

        Returns (NDArray[np.float32]):
            A numpy.ndarray object of shape (n, 3) containing the centers of the spheres composing
            the multisphere.
        """
        return self.centers

    def get_all_radii(self) -> NDArray[np.float32]:
        r"""Returns the radii of the spheres composing the multisphere.

        Returns (NDArray[np.float32]):
            A numpy.ndarray object of shape (n,) containing the radii of the spheres composing the
            multisphere.
        """
        return self.radii

    def get_all_centers_and_radii(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r"""Returns the centers and radii of the spheres composing the multisphere.

        Returns (tuple[NDArray[np.float32], NDArray[np.float32]]):
            A tuple containing two numpy.ndarray objects of shape (n, 3) and (n,) containing the
            centers and radii of the spheres composing the multisphere.
        """
        return self.get_all_centers(), self.get_all_radii()

    def __len__(self) -> int:
        r"""Returns the number of spheres composing the multisphere.

        Returns (int):
            The number of spheres composing the multisphere.
        """
        return self.centers.shape[0]
