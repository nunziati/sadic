r"""API functions for execution of sadic algorithm on a protein."""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from sadic.solid import Solid, Sphere, Multisphere, VoxelSolid
from sadic.pdb import PDBEntity, Model
from sadic.algorithm.radius import find_max_radius_point, find_max_radius_point_voxel
from sadic.algorithm.depth import sadic_sphere as sadic_multisphere
from sadic.algorithm.depth import sadic_original_voxel as sadic_voxel
from sadic.utils import Repr


representation_options: dict[str, dict[str, Any]] = {
    "multisphere": {
        "solid_type": Multisphere,
        "probe_radius_function": find_max_radius_point,
        "depth_index_function": sadic_multisphere,
    },
    "voxel": {
        "solid_type": VoxelSolid,
        "probe_radius_function": find_max_radius_point_voxel,
        "depth_index_function": sadic_voxel,
    },
}


class SadicModelResult(Repr):
    r"""Result of the SADIC algorithm for a single model of a protein.

    Attributes:
        atom_index (NDArray[np.int32]):
            The index of the atoms of the protein.
        depth_index (NDArray[np.float32]):
            The SADIC depth index of the atoms of the protein.

    Methods:
        __init__:
            Initialize the result of the SADIC algorithm.
    """

    def __init__(self, atom_index: NDArray[np.int32], depth_index: NDArray[np.float32]) -> None:
        r"""Initialize the result of the SADIC algorithm.

        Args:
            atom_index (NDArray[np.int32]):
                The index of the atoms of the protein.
            depth_index (NDArray[np.float32]):
                The SADIC depth index of the atoms of the protein.
        """
        self.atom_index: NDArray[np.int32] = atom_index
        self.depth_index: NDArray[np.float32] = depth_index


class SadicEntityResult(Repr):
    r"""Result of the SADIC algorithm for an entity of a protein.

    Can be composed of multiple models.

    Attributes:
        result_list (list[SadicModelResult]):
            The list of results of the SADIC algorithm for each model of the protein.

    Methods:
        __init__:
            Initialize the result of the SADIC algorithm.
    """

    def __init__(self, result_list: list[SadicModelResult], sadic_args: dict[str, Any]) -> None:
        self.result_list: list[SadicModelResult] = result_list
        self.sadic_args: dict[str, Any] = sadic_args


def sadic(
    input_arg: str | PandasPdb | Structure,
    model_indexes: None | Sequence[int] = None,
    filter_arg: None
    | dict[str, str | int | Sequence[str] | Sequence[int]]
    | Sphere
    | NDArray[np.float32] = None,
    probe_radius: None | int | float = None,
    vdw_radii: None | dict[str, float] = None,
    representation: str = "voxel",
) -> SadicEntityResult:
    r"""Compute the SADIC depth index of a protein.

    Args:
        arg (str | PandasPdb | Structure):
            The protein to compute the SADIC depth index of. Can be a path to a PDB file, a
            PandasPdb object or a BioPython Structure object.
        model_indexes (None | Sequence[int]):
            The indexes of the models of the protein to compute the SADIC depth index of. If None,
            all the models are considered. Defaults to None.
        filter_arg (None | dict[str, str | int | Sequence[str] | Sequence[int]] | Sphere
        | NDArray[np.float32]):
            The filter to apply to the atoms of the protein. If None, no filter is applied. If a
            Sphere, only the atoms inside the sphere are considered. If a numpy array, it must be an
            array with shape (n_points, 3) containing the coordinates of the points to compute the
            SADIC depth index of. If a dictionary, the keys are the columns of the PDB file and the
            values are the values to select. The keys can be one of the following: "atom_number",
            "atom_name", "residue_name", "residue_number", "chain_id", "element_symbol". The values
            can be a single value or a list of values. Defaults to None.
        probe_radius (None | int | float):
            The radius of the probe to use to compute the SADIC depth index. If None, the optimal
            radius is computed for each model. If int, the radius is the same for all the models and
            it is equal to the optimal radius of the selected model. For consistency with the pdb
            notation, model indexes start from 1. If float, the radius is the same for all the
            models and it is equal to the value of the parameter. Defaults to None.
        vdw_radii (None | dict[str, float]):
            The van der Waals radii of the atoms of the protein. If None, the default values are
            used. Defaults to None.
        representation (str):
            Representation of the protein. Can be "multisphere" or "voxel". Defaults to "voxel".

    Returns (SadicEntityResult):
        The SADIC depth index of the atoms of the protein.
    """
    if representation not in representation_options:
        raise ValueError("Representation must be 'multisphere' or 'voxel'")

    print("Loading protein".ljust(30, "."), end="", flush=True)
    protein = PDBEntity(input_arg, vdw_radii=vdw_radii)
    print("DONE")

    # retrieve probe_radius
    original_probe_radius: None | float | int = probe_radius
    fixed_probe_radius: bool = False
    if probe_radius is not None:
        if not isinstance(probe_radius, int) and not isinstance(probe_radius, float):
            raise TypeError("Probe radius must be int, float or None")

        if isinstance(probe_radius, int):
            if probe_radius not in protein.models:
                raise ValueError(f"Model {probe_radius} not found in the protein")

            solid: Solid = representation_options[representation]["solid_type"](protein)
            probe_radius = representation_options[representation]["probe_radius_function"](solid)[1]
            fixed_probe_radius = True

        elif isinstance(probe_radius, float):
            if probe_radius <= 0:
                raise ValueError("Probe radius must be positive")

            fixed_probe_radius = True

    original_model_indexes: None | Sequence[int] = model_indexes
    model_indexes = model_indexes if model_indexes is not None else range(1, protein.nmodels + 1)
    results: list[SadicModelResult] = []
    for model_index in model_indexes:
        if model_index not in protein.models:
            raise ValueError(f"Model {model_index} not found in the protein")

        print("Creating solid".ljust(30, "."), end="", flush=True)
        solid: Solid = representation_options[representation]["solid_type"](
            protein.models[model_index]
        )
        print("DONE")

        if not fixed_probe_radius:
            print(
                f"Computing max radius for model {model_index}".ljust(30, "."), end="", flush=True
            )
            probe_radius = representation_options[representation]["probe_radius_function"](solid)[1]
            print("DONE")
            print(f"probe radius: {probe_radius}")

        print(f"Computing depth indexes for model {model_index}".ljust(30, "."), end="", flush=True)
        filtered_model: Model = (
            protein.models[model_index]
            if filter_arg is None
            else protein.models[model_index].filter(filter_arg)
        )
        results.append(
            SadicModelResult(
                protein.models[model_index].atom_indexes,
                representation_options[representation]["depth_index_function"](
                    solid, filtered_model, probe_radius
                )[0],
            )
        )
        print("DONE")

    sadic_args: dict[str, Any] = {
        "representation": representation,
        "probe_radius": original_probe_radius,
        "filter": filter_arg,
        "vdw_radii": vdw_radii,
        "model_indexes": original_model_indexes,
    }

    return SadicEntityResult(results, sadic_args)
