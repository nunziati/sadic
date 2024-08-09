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
DEFAULT_METHOD = "basic_vectorized"
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

    print_task("Finding reference radius")
    time3 = time.time()
    reference_radius, complexity_variables_3 = find_reference_radius(reference_radius_method, solid, atoms, extreme_coordinates=extreme_coordinates, resolution=resolution)

    print_task("Computing indexes method 1")
    time4 = time.time()
    result_1, complexity_variables_4_1 = compute_indexes(indexes_computation_method, solid, atoms, reference_radius, extreme_coordinates=extreme_coordinates, resolution=resolution)
    
    time5_a = time.time()
    print_task("Computing indexes method 2")
    result_2, complexity_variables_4_2 = compute_indexes("translated_sphere_vectorized", solid, atoms, reference_radius, extreme_coordinates=extreme_coordinates, resolution=resolution)
    print_task()

    time5_b = time.time()

    print("Discretization time: ", time2 - time1)
    print("Remove holes time: ", time3 - time2)
    print("Find reference radius time: ", time4 - time3)
    print("Compute indexes method 1 time: ", time5_a - time4)
    print("Compute indexes method 2 time: ", time5_b - time5_a)



    diff = np.abs(result_1 - result_2)
    rel_diff = diff / np.abs(result_1)

    print("Check positive sign: ", np.all(result_1 >= 0))
    print("Check positive sign: ", np.all(result_2 >= 0))
    # plot histogram
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.hist(diff, bins=100)
    plt.figure(2)
    plt.hist(rel_diff, bins=100)
    plt.show()



    return dict(
        result = result_1,
        times = (time2 - time1, time3 - time2, time4 - time3, time5_a - time4),
        complexity_variables = {
            "N": atoms.shape[0],
            "n": solid.shape[0] * solid.shape[1] * solid.shape[2],
            "1": complexity_variables_1,
            "2": complexity_variables_2,
            "3": complexity_variables_3,
            "4": complexity_variables_4_1
        }
    )

def main():
    args = parse_args()
    
    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose

    output = process_protein(input_arg, resolution=resolution, method=method, verbose=verbose)
    
if __name__ == "__main__":
    main()