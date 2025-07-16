import time
import argparse
import os

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

DEFAULT_INPUT = "4ins"
DEFAULT_RESOLUTION = 0.5
DEFAULT_METHOD = None
DEFAULT_VERBOSE = True

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    return parser.parse_args()

def plot_3d_points(voxel_centers):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers.astype(np.float64))
    # Optional: color all voxels the same (e.g., blue)
    # Normalize z values to [0, 1] for coloring
    z_values = voxel_centers[:, 2].astype(np.float32)
    z_min, z_max = z_values.min(), z_values.max()
    if z_max > z_min:
        normalized_z = (z_values - z_min) / (z_max - z_min)
    else:
        normalized_z = np.zeros_like(z_values)
    # Map normalized z to a color gradient (e.g., blue to red)
    colors = np.zeros((voxel_centers.shape[0], 3))
    colors[:, 0] = normalized_z        # Red increases with z
    colors[:, 2] = 1 - normalized_z    # Blue decreases with z
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def process_protein(input_arg, vdw_radii = None, resolution = 0.3, method=None, verbose=True, single_methods=None, atom_refs=None, atom_refs_file=None):
    if single_methods is None:
        single_methods = dict(
            alignment_method = "approximate",
            discretization_method = "basic_vectorized_0.5" if method is None else method,
            fill_space_method = "skimage_ball_0.5" if method is None else method,
            holes_removal_method = "basic_vectorized" if method is None else method,
            reference_radius_method = "basic_vectorized" if method is None else method,
            indexes_computation_method = "translated_sphere_vectorized_0.5" # basic_vectorized" # "kd_trees_scipy_tree_comparison" # kd_trees_scipy_tree_comparison" # "kd_trees_scipy_tree_comparison translated_sphere_vectorized"
        )
    
    alignment_method = single_methods["alignment_method"]
    discretization_method = single_methods["discretization_method"]
    fill_space_method = single_methods["fill_space_method"]
    holes_removal_method = single_methods["holes_removal_method"]
    reference_radius_method = single_methods["reference_radius_method"]
    indexes_computation_method = single_methods["indexes_computation_method"]

    
    print_task = TaskPrinter(verbose=verbose)
    print_task("Loading protein")
    if atom_refs is not None:
        protein = PDBEntity(input_arg, vdw_radii=vdw_radii, atom_refs=atom_refs)
    elif atom_refs_file is not None:
        protein = PDBEntity(input_arg, vdw_radii=vdw_radii, atom_refs_file=atom_refs_file)
    else:
        protein = PDBEntity(input_arg, vdw_radii=vdw_radii)
    
    print_task("Preliminaries")
    model = protein.models[1]

    atoms = model.get_centers()

    # plot_3d_points(atoms)
    # input()
    
    if fill_space_method == "none":
        radii = model.get_radii(probe_radius = None)
    else:
        radii = model.get_radii(probe_radius = 0.)

    print_task("Alignment")
    time_align_start = time.time()
    atoms, _, complexity_variables_alignment = align(alignment_method, atoms)
    time_align_end = time.time()

    section = False

    if section:
        # CAMERA PARAMETERS FROM RCSB
        position = np.array([50.63, -25.4, 22.28])
        target   = np.array([0.34, 5.34, 16.22])
        up       = np.array([-0.33, -0.67, -0.67])

        # --- 1. Build Camera Coordinate Frame ---
        # z_c: camera's forward direction (from position to target)
        z_c = target - position
        z_c = z_c / np.linalg.norm(z_c)

        # x_c: right vector (cross up and direction)
        x_c = np.cross(up, z_c)
        x_c = x_c / np.linalg.norm(x_c)

        # y_c: "true up" (orthogonal to z_c and x_c)
        y_c = np.cross(z_c, x_c)
        y_c = y_c / np.linalg.norm(y_c)

        # Rotation matrix: columns are x_c, y_c, z_c (camera frame axes in world coordinates)
        R = np.vstack([x_c, y_c, z_c]).T  # shape (3, 3)

        # --- 2. Transform Points into Camera-Aligned Frame ---
        # First, translate so camera is at the origin:
        atoms = atoms - position

        # Now rotate:
        atoms = atoms @ R
        atoms[:,2] *= -1
        atoms[:,0] *= -1

        # Rotation matrices
        deg2rad = np.pi / 180

        theta_x = 60 * deg2rad
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x),  np.cos(theta_x)]
        ])

        theta_z = 30 * deg2rad
        R_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z),  np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # Apply rotation: first Rx, then Rz
        R = R_z @ R_x
        atoms = atoms @ R.T

    print_task("Computing extreme coordinates")
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
    discretization_time_start = time.time()
    solid, complexity_variables_discretization = discretize(discretization_method, model, extreme_coordinates=extreme_coordinates, resolution=resolution)
    discretization_time_end = time.time()

    section = False

    if section:
        return dict(
            protein = protein.models[1],
            complexity_variables = {
                "N": atoms.shape[0],
                "n": solid.shape[0] * solid.shape[1] * solid.shape[2],
            },
        )

    section = False
    if section:
        def plot_solid(input_solid, filename):
            print_task("Extracting outer surface")
            surface_extraction_time_start = time.time()
            padded_solid = np.pad(input_solid, pad_width=1, mode='constant', constant_values=0)
            surface = np.zeros_like(padded_solid, dtype=bool)

            for x in range(1, padded_solid.shape[0] - 1):
                for y in range(1, padded_solid.shape[1] - 1):
                    for z in range(1, padded_solid.shape[2] - 1):
                        if padded_solid[x, y, z] and not np.all(padded_solid[x-1:x+2, y-1:y+2, z-1:z+2]):
                            surface[x, y, z] = True
            surface_extraction_time_end = time.time()
            print_task(f"Surface extraction completed in {surface_extraction_time_end - surface_extraction_time_start:.2f} seconds")

            print_task("Plotting surface")

            # Create figure and 3D axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Set voxel colors (all blue)
            colors = np.empty(surface.shape, dtype=object)
            colors[surface] = 'lightblue'
            
            # Plot voxels
            ax.voxels(surface, facecolors=colors, edgecolor='black', linewidth=0.5)
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlim(0, 35)
            ax.set_ylim(0, 35)
            ax.set_zlim(0, 35)


            # Save the figure in pdf
            plt.savefig(filename, bbox_inches='tight', pad_inches=-0.6)
            plt.show()

    print_task("Space filling")
    space_fill_time_start = time.time()
    if fill_space_method != "none":
        print("Fillo")
        solid, complexity_variables_space_filling = fill_space(fill_space_method, solid, resolution=resolution, probe_radius=PDBEntity.vdw_radii['O'])
    else:
        complexity_variables_space_filling = dict(protein_int_volume=complexity_variables_discretization["protein_int_volume"])
    space_fill_time_end = time.time()

    if section:
        plot_solid(solid, "solid_spacefill.pdf")

    # Visualize the resulting voxels with open3d
    voxel_centers = np.argwhere(solid)
    # plot_3d_points(voxel_centers)

    # input()

    print_task("Removing holes")
    holes_removal_time_start = time.time()
    solid, complexity_variables_holes_removal = remove_holes(holes_removal_method, solid)
    holes_removal_time_end = time.time()

    print_task("Finding reference radius")
    reference_radius_time_start = time.time()
    reference_radius, complexity_variables_reference_radius = find_reference_radius(reference_radius_method, solid, atoms, extreme_coordinates=extreme_coordinates, resolution=resolution)
    reference_radius_time_end = time.time()
    print_task("Computing indexes")
    
    compute_indexes_time_start = time.time()
    result, complexity_variables_depth_indexes = compute_indexes(indexes_computation_method, solid, atoms, reference_radius, extreme_coordinates=extreme_coordinates, resolution=resolution)
    compute_indexes_time_end = time.time()
    print_task()
    
    return dict(
        protein = protein.models[1],
        result = result,
        times = dict(
            alignment = time_align_end - time_align_start,
            discretization = discretization_time_end - discretization_time_start,
            space_fill = space_fill_time_end - space_fill_time_start,
            holes_removal = holes_removal_time_end - holes_removal_time_start,
            reference_radius = reference_radius_time_end - reference_radius_time_start,
            compute_indexes = compute_indexes_time_end - compute_indexes_time_start
        ),
        complexity_variables = {
            "N": atoms.shape[0],
            "n": solid.shape[0] * solid.shape[1] * solid.shape[2],
            "alignment": complexity_variables_alignment,
            "discretization": complexity_variables_discretization,
            "space_filling": complexity_variables_space_filling,
            "holes_removal": complexity_variables_holes_removal,
            "reference_radius": complexity_variables_reference_radius,
            "indexes_computation": complexity_variables_depth_indexes
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

    depth_indexes_path = os.path.join("new_experiments", "single_sadic", "depth_indexes", f"{input_arg.strip()}.npy")
    os.makedirs(os.path.dirname(depth_indexes_path), exist_ok=True)
    np.save(depth_indexes_path, output["result"])

    return None

    print(output["times"]) 
    print("n:", output["complexity_variables"]["n"])
    print("N:", output["complexity_variables"]["N"])
    print("Solid volume", np.sum(output["solid"]))
    print("Reference radius:", output["reference_radius"])

    print(output["complexity_variables"]["discretization"]["protein_int_volume"])
    print(output["complexity_variables"]["space_filling"]["protein_int_volume"])
    print(output["complexity_variables"]["holes_removal"]["n_components"])
    print(output["complexity_variables"]["holes_removal"]["protein_int_volume"])

    image = output["complexity_variables"]["4"]["voxel_operations_map"]
    p_list = output["complexity_variables"]["4"]["p_list"]
    p_max = np.max(p_list)
    N_p_avg = np.mean(p_list) * output["complexity_variables"]["N"]
    print("Max p:", p_max)
    print("N*p_max:", output["complexity_variables"]["N"] * p_max)
    print("N*p_avg:", N_p_avg)

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
