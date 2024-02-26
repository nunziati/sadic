r"""API functions for execution of sadic algorithm on a protein."""

from typing import Any, Sequence
import time

import numpy as np
from numpy.typing import NDArray
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure

from sadic.solid import Solid, Multisphere, VoxelSolid
from sadic.pdb import PDBEntity, Model, SadicModelResult, SadicEntityResult
from sadic.algorithm.radius import find_max_radius_point, find_max_radius_point_voxel
from sadic.algorithm.depth import sadic_sphere as sadic_multisphere
from sadic.algorithm.depth import sadic_original_voxel as sadic_voxel


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


def sadic(
    input_arg: str | PandasPdb | Structure,
    input_mode: str = 'infer',
    model_indexes: None | Sequence[int] = None,
    filter_by: None
    | dict[str, str | int | Sequence[str] | Sequence[int]]
    | tuple[NDArray[np.float32], float]
    | NDArray[np.float32] = None,
    probe_radius: None | int | float = None,
    vdw_radii: None | dict[str, float] = None,
    representation: str = "voxel",
    resolution: None | int | float = 0.3,
    debug: bool = False,
) -> SadicEntityResult:
    r"""Compute the SADIC depth index of a protein.

    Args:
        arg (str | PandasPdb | Structure):
            The protein to compute the SADIC depth index of. Can be a path to a PDB file, a
            PandasPdb object or a BioPython Structure object.
        input_mode (str, optional):
            The mode to be used to build the PDBEntity object. Can be one of "biopandas",
            "biopython", "pdb", "gz", "url", "code" or "infer". Defaults to "infer".
        model_indexes (None | Sequence[int]):
            The indexes of the models of the protein to compute the SADIC depth index of. If None,
            all the models are considered. Defaults to None.
        filter_by (None | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float] | NDArray[np.float32]):
            The filter to apply to the atoms of the protein. If None, no filter is applied. If a
            tuple of a np.ndarray and a float, they are assumed to represent the center and radius
            of a sphere and only the atoms inside the sphere are considered. If a numpy array, it
            must be an array with shape (n_points, 3) containing the coordinates of the points to be
            selected. If a dictionary, the keys are the columns of the PDB file and the values are
            the values to select. The keys can be one of the following: "atom_number", "atom_name",
            "residue_name", "residue_number", "chain_id", "element_symbol". The values can be a
            single value or a list of values. Defaults to None.
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
        resolution (None | float | str):
            Resolution of the voxel representation of the protein. If None, the resolution is
            set to 0.3 Angstrom. If float, the resolution is set to the value of the parameter. If
            str, it must be equal to 'old' and the resolution is set to 1.0 Angstrom, that is an
            equivalent resolution to the one used in the original SADIC implementation. Defaults to
            None.
        debug (bool):
            If True, the algorithm prints and returns debug information. Defaults to False.

    Returns (SadicEntityResult):
        The SADIC depth index of the atoms of the protein. If debug is True, it also returns debug
        information in the form of a dictionary containing the grid_shape.
    """
    if representation not in representation_options:
        raise ValueError("Representation must be 'multisphere' or 'voxel'")

    if representation == "voxel":
        if resolution is None:
            resolution = 0.3
    
        if resolution is not None and isinstance(resolution, float) and resolution <= 0:
            raise ValueError("Resolution must be positive")
        
        if resolution is not None and isinstance(resolution, str):
            if resolution != "old":
                raise ValueError("Resolution must be 'old' or positive float")
            resolution = 1.0
    
    print("Loading protein".ljust(30, "."), end="", flush=True)
    protein = PDBEntity(input_arg, vdw_radii=vdw_radii)
    print("DONE")

    start_time = time.time()

    # retrieve probe_radius
    original_probe_radius: None | float | int = probe_radius
    fixed_probe_radius: bool = False
    if probe_radius is not None:
        if not isinstance(probe_radius, int) and not isinstance(probe_radius, float):
            raise TypeError("Probe radius must be int, float or None")

        if isinstance(probe_radius, int):
            if probe_radius not in protein.models:
                raise ValueError(f"Model {probe_radius} not found in the protein")

            solid: Solid = representation_options[representation]["solid_type"](protein, resolution=resolution)
            probe_radius = representation_options[representation]["probe_radius_function"](solid)[1]
            fixed_probe_radius = True

        elif isinstance(probe_radius, float):
            if probe_radius <= 0:
                raise ValueError("Probe radius must be positive")

            fixed_probe_radius = True

    original_model_indexes: None | Sequence[int] = model_indexes
    model_indexes = model_indexes if model_indexes is not None else range(1, protein.nmodels + 1)
    results: dict[int, SadicModelResult] = {}
    for model_index in model_indexes:
        if model_index not in protein.models:
            raise ValueError(f"Model {model_index} not found in the protein")

        print("Creating solid".ljust(30, "."), end="", flush=True)
        solid: Solid = representation_options[representation]["solid_type"](
            protein.models[model_index], resolution=resolution
        ).remove_holes()
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
            if filter_by is None
            else protein.models[model_index].filter(filter_by)
        )
        results[model_index] = SadicModelResult(
            protein.models[model_index].atom_indexes,
            representation_options[representation]["depth_index_function"](
                solid, filtered_model, probe_radius
            )[0],
            filtered_model,
        )
        print("DONE")

    sadic_args: dict[str, Any] = {
        "representation": representation,
        "probe_radius": original_probe_radius,
        "filter": filter_by,
        "vdw_radii": vdw_radii,
        "model_indexes": original_model_indexes,
    }

    if debug:
        debug_dict = {
            "grid_shape": solid.dimensions,
            "execution_time": time.time() - start_time,
        }

        print("Debug info")
        print(debug_dict)

        return SadicEntityResult(results, sadic_args, protein), debug_dict

    return SadicEntityResult(results, sadic_args, protein)
