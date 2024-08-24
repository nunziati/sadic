import time
import argparse

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sadic.pdb import PDBEntity

from algorithms.align import align
from algorithms.discretize import discretize
from algorithms.fill_space import fill_space
from algorithms.remove_holes import remove_holes
from algorithms.reference_radius import find_reference_radius
from algorithms.compute_indexes import compute_indexes

from mylogging import TaskPrinter

DEFAULT_INPUT = "1ubq"
DEFAULT_RESOLUTION = 0.5
DEFAULT_METHOD = None # "basic_vectorized"
DEFAULT_VERBOSE = True

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    return parser.parse_args()

def process_protein(input_arg, vdw_radii = None, resolution = 0.3, method=None, verbose=True):
    alignment_method = "basic"
    discretization_method = "basic_vectorized" if method is None else method
    fill_space_method = "none"
    holes_removal_method = "basic_vectorized" if method is None else method
    reference_radius_method = "basic_vectorized" if method is None else method
    indexes_computation_method = "translated_sphere_vectorized"
    # indexes_computation_method = "octrees_range_query"
    # indexes_computation_method = "kdtrees_sklearn"
    # indexes_computation_method = "kdtrees_voxel_scan_loop"
    
    print_task = TaskPrinter(verbose=verbose)
    print_task("Loading protein")
    protein = PDBEntity(input_arg, vdw_radii=vdw_radii)
    
    print_task("Preliminaries")
    model = protein.models[1]

    atoms = model.get_centers()
    
    if fill_space_method == "none":
        radii = model.get_radii(probe_radius = None)
    else:
        radii = model.get_radii(probe_radius = 0.)

    print_task("Alignment")
    time_align_start = time.time()
    atoms, _ = align(alignment_method, atoms)
    time_align_end = time.time()

    print_task("Computing extreme coordinates")
    time_computing_extreme_coordinates_start = time.time()
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
    time_computing_extreme_coordinates_end = time.time()

    model = dict(atoms=atoms, radii=radii)

    print_task("Discretization")
    time1 = time.time()
    solid, complexity_variables_1 = discretize(discretization_method, model, extreme_coordinates=extreme_coordinates, resolution=resolution)

    print_task("Space filling")
    time1_b = time.time()
    if fill_space_method != "none":
        solid, complexity_variables_1b = fill_space(discretization_method, solid, resolution=resolution, probe_radius=PDBEntity.vdw_radii['O'])
    else:
        complexity_variables_1b = dict()

    print_task("Removing holes")
    time2 = time.time()
    solid, complexity_variables_2 = remove_holes(holes_removal_method, solid)

    print_task("Finding reference radius")
    time3 = time.time()
    reference_radius, complexity_variables_3 = find_reference_radius(reference_radius_method, solid, atoms, extreme_coordinates=extreme_coordinates, resolution=resolution)

    print_task("Computing indexes")
    time4 = time.time()
    result, complexity_variables_4 = compute_indexes(indexes_computation_method, solid, atoms, reference_radius, extreme_coordinates=extreme_coordinates, resolution=resolution)

    print_task()

    time5 = time.time()

    return dict(
        result = result,
        times = (time_align_end - time_align_start, time1_b - time1, time2 - time1_b, time3 - time2, time4 - time3, time5 - time4),
        complexity_variables = {
            "N": atoms.shape[0],
            "n": solid.shape[0] * solid.shape[1] * solid.shape[2],
            "1": complexity_variables_1,
            "2": complexity_variables_2,
            "3": complexity_variables_3,
            "4": complexity_variables_4
        },
        reference_radius=reference_radius,
        solid=solid
    )

def main():
    args = parse_args()
    
    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose

    output = process_protein(input_arg, resolution=resolution, method=method, verbose=verbose)

    print(output["times"]) 
    print("n:", output["complexity_variables"]["n"])
    print("N:", output["complexity_variables"]["N"])
    print("Solid volume", np.sum(output["solid"]))
    print("Reference radius:", output["reference_radius"])
    image = output["complexity_variables"]["4"]["voxel_operations_map"]
    p_list = output["complexity_variables"]["4"]["p_list"]
    p_max = np.max(p_list)
    N_p_avg = np.mean(p_list) * output["complexity_variables"]["N"]
    print("Max p:", p_max)
    print("N*p_max:", output["complexity_variables"]["N"] * p_max)
    print("N*p_avg:", N_p_avg)

    # Average è esprimibile come percentuale del numero di atomi?
    # N*p_max/n è una costante?
    # Average è una costante?
    
    voxel_centers = np.argwhere(image)

    values = image[voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]]

    print("Number of visited voxels:", len(values))
    print("Max:", values.max(), "Min:", values.min())
    print("Average:", values.mean(), "Std:", values.std())

    plt.hist(values, bins=219)
    plt.show()

    # Normalize the values to be between 0 and 1
    normalized_values = (values - values.min()) / (values.max() - values.min())

    # Map normalized values to colors (from blue to red)
    colors = np.zeros((voxel_centers.shape[0], 3))
    colors[:, 0] = normalized_values  # Red channel
    colors[:, 2] = 1 - normalized_values  # Blue channel

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()