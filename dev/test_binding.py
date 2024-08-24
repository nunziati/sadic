import time
import argparse

import numpy as np

from sadic.pdb import PDBEntity

from algorithms.discretize import discretize
from algorithms.remove_holes import remove_holes
from algorithms.reference_radius import find_reference_radius
from algorithms.compute_indexes import compute_indexes

from mylogging import TaskPrinter

DEFAULT_INPUT = "1ubq"
DEFAULT_RESOLUTION = 0.5
DEFAULT_METHOD = "translated_sphere_vectorized"
DEFAULT_VERBOSE = True

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    return parser.parse_args()

def process_protein(input_arg, vdw_radii = None, resolution = 0.3, method=None, verbose=True):
    discretization_method = "basic_vectorized" if method is None else method
    holes_removal_method = "basic_vectorized" if method is None else method
    reference_radius_method = "basic_vectorized" if method is None else method
    indexes_computation_method = "basic_vectorized" if method is None else method

    print_task = TaskPrinter(verbose=verbose)
    print_task("Loading protein")
    protein = PDBEntity(input_arg, vdw_radii=vdw_radii)
    
    print_task("Preliminaries")
    model = protein.models[1]

    atoms = model.get_centers()
    radii = model.get_radii(probe_radius = None)

    extreme_coordinates = np.empty((3, 2), dtype=np.float32)
    for axis in range(3):
        extreme_coordinates[axis, 0] = np.min(atoms[:, axis] - radii)
        extreme_coordinates[axis, 1] = np.max(atoms[:, axis] + radii)

    extreme_coordinates[:, 1] += (
        resolution
        * (1 - np.modf(
            (extreme_coordinates[:, 1] - extreme_coordinates[:, 0])
            / resolution
        )[0])
    )

    model = dict(atoms=atoms, radii=radii)

    print_task("Discretization")
    time1 = time.time()
    solid, complexity_variables_1 = discretize(discretization_method, model, extreme_coordinates=extreme_coordinates, resolution=resolution)

    print_task("Removing holes")
    time2 = time.time()
    solid, complexity_variables_2 = remove_holes(holes_removal_method, solid)

    print_task("Finding reference radius method real")
    time3 = time.time()
    reference_radius_real, complexity_variables_3_real = find_reference_radius(reference_radius_method, solid, atoms, extreme_coordinates=extreme_coordinates, resolution=resolution)
    
    print_task("Finding reference radius method new")
    reference_radius_new, complexity_variables_3_new = find_reference_radius("coeurjolly_translated_sphere", solid, atoms, extreme_coordinates=extreme_coordinates, resolution=resolution)
    
    print_task("Computing indexes")
    time4 = time.time()
    result, complexity_variables_4 = compute_indexes(indexes_computation_method, solid, atoms, reference_radius_real, extreme_coordinates=extreme_coordinates, resolution=resolution)
    
    print_task()

    time5 = time.time()

    print(f"Reference radius real: {reference_radius_real}")
    print(f"Reference radius new: {reference_radius_new}")

    return dict(
        result = result,
        times = (time2 - time1, time3 - time2, time4 - time3, time5 - time4),
        complexity_variables = {
            "N": atoms.shape[0],
            "n": solid.shape[0] * solid.shape[1] * solid.shape[2],
            "1": complexity_variables_1,
            "2": complexity_variables_2,
            "3": complexity_variables_3,
            "4": complexity_variables_4
        }
    )

def main():
    args = parse_args()
    
    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose

    output = process_protein(input_arg, resolution=resolution, method=method, verbose=verbose)

    print(output["result"])
    print(output["times"])
    print(output["complexity_variables"])

if __name__ == "__main__":
    main()