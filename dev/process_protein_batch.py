import argparse
import csv
import random
import multiprocessing as mp
import queue
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from process_single_protein import process_protein

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DEFAULT_INPUT = "protein_sample.txt"
DEFAULT_EXPERIMENT_FOLDER = "experiments"
DEFAULT_EXPERIMENT_NAME = "aligned_spacefill_translated_sphere"
DEFAULT_RESOLUTION = 0.5
DEFAULT_METHOD = "translated_sphere_vectorized"
DEFAULT_VERBOSE = False
DEFAULT_SUBSET = 1000
DEFAULT_UNIFORM = True
DEFAULT_RESUME = -1
DEFAULT_NUM_PROCESSES = 8


############################## CONFIGURE METHODS ##############################
ALIGNMENT_METHOD = "basic"
DISCRETIZATION_METHOD = "basic_vectorized"
FILL_SPACE_METHOD = "skimage"
HOLES_REMOVAL_METHOD = "basic_vectorized"
REFERENCE_RADIUS_METHOD = "basic_vectorized"
INDEXES_COMPUTATION_METHOD = "translated_sphere_vectorized"
###############################################################################



OUTPUT_FILE_HEADER = (
    "PDB_ID",
    "resolution",
    "alignment_method",
    "discretization_method",
    "fill_space_method",
    "holes_removal_method",
    "reference_radius_method",
    "indexes_computation_method",
    "t_alignment",
    "t_discretization",
    "t_fill_space",
    "t_holes_removal",
    "t_reference_radius",
    "t_indexes_computation",
    "N",
    "n",
    "reference_radius",
    "protein_int_volume_discretization",
    "protein_int_volume_space_fill",
    "connected_components",
    "protein_int_volume_holes_removal",
    "p_disc_min",
    "p_disc_max",
    "p_disc_avg",
    "p_disc_med",
    "p_disc_std",
    "p_idx_min",
    "p_idx_max",
    "p_idx_avg",
    "p_idx_med",
    "p_idx_std",
    "N*p_idx_max/n",
    "N*p_idx_avg/n",
    "idx_visited_voxels",
    "max_idx_visited_voxel_map",
    "min_idx_visited_voxel_map",
    "avg_idx_visited_voxel_map",
    "std_idx_visited_voxel_map"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("--experiment_folder", type=str, default=DEFAULT_EXPERIMENT_FOLDER, help="Output experiment folder")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Output experiment name")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET, help="Number of proteins to process")
    parser.add_argument("--uniform", action="store_true", default=DEFAULT_UNIFORM, help="Sample uniformly on the number of atoms")
    parser.add_argument("--resume", type=int, default=DEFAULT_RESUME, help="Index of the protein to resume from")
    parser.add_argument("--num_processes", type=int, default=DEFAULT_NUM_PROCESSES, help="Number of processes to use for parallel processing")
    return parser.parse_args()

def process_single_protein_and_extract_output(pdb_id, resolution=0.3, method=None, experiment_folder="", verbose=True):
    if experiment_folder == "":
        raise ValueError("experiment_folder must be specified")
    
    try:
        output = process_protein(pdb_id.strip(), resolution=resolution, method=method, verbose=verbose,
                                 single_methods={
                                        "alignment_method": ALIGNMENT_METHOD,
                                        "discretization_method": DISCRETIZATION_METHOD,
                                        "fill_space_method": FILL_SPACE_METHOD,
                                        "holes_removal_method": HOLES_REMOVAL_METHOD,
                                        "reference_radius_method": REFERENCE_RADIUS_METHOD,
                                        "indexes_computation_method": INDEXES_COMPUTATION_METHOD
                                 })
    except KeyboardInterrupt as e:
        print("Interrupted by the user\n", end="\n")
        raise e
    except:
        print(f"Error processing protein {pdb_id.strip()}\n", end="\n")
        return tuple([pdb_id.strip(), "ERROR"] + [None] * (len(OUTPUT_FILE_HEADER) - 2))

    depth_indexes_path = os.path.join(experiment_folder, "depth_indexes", f"{pdb_id.strip()}.npy")
    p_disc_path = os.path.join(experiment_folder, "discretization", "p", f"{pdb_id.strip()}.npy")
    disc_voxel_operations_map_path = os.path.join(experiment_folder, "discretization", "voxel_operations_map", f"{pdb_id.strip()}.npy")
    p_idx_path = os.path.join(experiment_folder, "indexes_computation", "p", f"{pdb_id.strip()}.npy")
    idx_voxel_operations_map_path = os.path.join(experiment_folder, "indexes_computation", "voxel_operations_map", f"{pdb_id.strip()}.npy")

    if not os.path.exists(os.path.join(experiment_folder, "depth_indexes")):
        os.makedirs(os.path.join(experiment_folder, "depth_indexes"))
    if not os.path.exists(os.path.join(experiment_folder, "discretization", "p")):
        os.makedirs(os.path.join(experiment_folder, "discretization", "p"))
    if not os.path.exists(os.path.join(experiment_folder, "discretization", "voxel_operations_map")):
        os.makedirs(os.path.join(experiment_folder, "discretization", "voxel_operations_map"))
    if not os.path.exists(os.path.join(experiment_folder, "indexes_computation", "p")):
        os.makedirs(os.path.join(experiment_folder, "indexes_computation", "p"))
    if not os.path.exists(os.path.join(experiment_folder, "indexes_computation", "voxel_operations_map")):
        os.makedirs(os.path.join(experiment_folder, "indexes_computation", "voxel_operations_map"))

    p_disc_array = np.array(output["complexity_variables"]["discretization"]["p_list"], dtype=np.int32)
    p_idx_array = np.array(output["complexity_variables"]["indexes_computation"]["p_list"], dtype=np.int32)

    disc_voxel_operations_map = np.array(output["complexity_variables"]["discretization"]["visit_map"], dtype=np.int32)
    idx_voxel_operations_map = np.array(output["complexity_variables"]["indexes_computation"]["voxel_operations_map"], dtype=np.int32)
    
    np.save(depth_indexes_path, output["result"])
    np.save(p_disc_path, p_disc_array)
    np.save(disc_voxel_operations_map_path, disc_voxel_operations_map)
    np.save(p_idx_path, p_idx_array)
    np.save(idx_voxel_operations_map_path, idx_voxel_operations_map)

    idx_voxel_centers = np.argwhere(idx_voxel_operations_map)
    values = idx_voxel_operations_map[idx_voxel_centers[:, 0], idx_voxel_centers[:, 1], idx_voxel_centers[:, 2]]

    output_tuple = (
        pdb_id.strip(),
        resolution,
        ALIGNMENT_METHOD,
        DISCRETIZATION_METHOD,
        FILL_SPACE_METHOD,
        HOLES_REMOVAL_METHOD,
        REFERENCE_RADIUS_METHOD,
        INDEXES_COMPUTATION_METHOD,
        output["times"]["alignment"],
        output["times"]["discretization"],
        output["times"]["space_fill"],
        output["times"]["holes_removal"],
        output["times"]["reference_radius"],
        output["times"]["compute_indexes"],
        output["complexity_variables"]["N"],
        output["complexity_variables"]["n"],
        output["reference_radius"],
        output["complexity_variables"]["discretization"]["protein_int_volume"],
        output["complexity_variables"]["space_filling"]["protein_int_volume"],
        output["complexity_variables"]["holes_removal"]["n_components"],
        output["complexity_variables"]["holes_removal"]["protein_int_volume"],
        p_disc_array.min(),
        p_disc_array.max(),
        p_disc_array.mean(),
        np.median(p_disc_array),
        p_disc_array.std(),
        p_idx_array.min(),
        p_idx_array.max(),
        p_idx_array.mean(),
        np.median(p_idx_array),
        p_idx_array.std(),
        output["complexity_variables"]["N"] * p_idx_array.max() / output["complexity_variables"]["n"],
        output["complexity_variables"]["N"] * p_idx_array.mean() / output["complexity_variables"]["n"],
        len(values),
        values.max(),
        values.min(),
        values.mean(),
        values.std()
    )
    
    return output_tuple

def process_protein_batch(pdb_ids, resolution=0.3, method=None, experiment_folder="", verbose=True):
    if experiment_folder == "":
        raise ValueError("experiment_folder must be specified")
    # prepare the output_file as a list of tuples
    output_file = []
    
    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, experiment_folder=experiment_folder, verbose=verbose)

        output_file.append(output_tuple)

    return output_file

def pick_uniform_tuples(tuples_list, N):
    # Sort the tuples by the second element
    sorted_tuples = sorted(set(tuples_list), key=lambda x: x[1])

    # Extract the second elements and calculate the empirical CDF
    second_elements = np.array([y for _, y in sorted_tuples])

    min_y = second_elements[0]
    max_y = second_elements[-1]
    
    # Generate N random numbers uniformly distributed between min_y and max_y
    random_second_element = np.random.uniform(min_y, max_y, N)

    # Find the indices of the closest elements in the sorted list, removing the picked elements
    indices = []
    for y in random_second_element:
        idx = np.argmin(np.abs(second_elements - y))
        indices.append(idx)
        second_elements = np.delete(second_elements, idx)
    
    # Select the tuples corresponding to these indices
    selected_tuples = [sorted_tuples[i] for i in indices]
    
    return selected_tuples

def sublist_worker(sublist, resolution, method, experiment_folder, verbose):
    return process_protein_batch(sublist, resolution, method, experiment_folder, verbose)

def queue_worker(input_queue, output_queue, n_proteins, resolution, method, experiment_folder, verbose):
    while not input_queue.empty():
        try:
            pdb_id, idx = input_queue.get_nowait()
        except queue.Empty as e:
            print(e)
            break
        except Exception as e:
            # create an "exception" file
            with open("exception.txt", "w") as f:
                f.write("Error processing protein " + pdb_id + "with idx" + str(idx) + "\n")
                f.write(str(e))
            raise e

        print(f"Processing protein {idx + 1}/{n_proteins}\n", end="\n")
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, experiment_folder=experiment_folder, verbose=verbose)
        output_queue.put((idx, output_tuple))
        print("Queue worker finished", idx + 1, end="\n")

def process_protein_batch_scalar(pdb_ids, resolution, method, experiment_folder, verbose):
    output_file = [OUTPUT_FILE_HEADER]

    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, experiment_folder=experiment_folder, verbose=verbose)

        output_file.append(output_tuple)

    return output_file


def process_protein_batch_in_parallel_sublists(pdb_ids, resolution, method, verbose, experiment_folder, num_processes):
    # Split the pdb_ids into sublists for each process
    sublists = [pdb_ids[i::num_processes] for i in range(num_processes)]
    
    # Create a multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the worker function
        results = pool.starmap(sublist_worker, [(sublist, resolution, method, experiment_folder, verbose) for sublist in sublists])
    
    # Merge the results from all processes
    merged_results = [item for sublist in results for item in sublist]
    
    output_file = [OUTPUT_FILE_HEADER]
    output_file += merged_results

    return output_file

def process_protein_batch_in_parallel_queue(pdb_ids, resolution, method, verbose, experiment_folder, num_processes):
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    n_proteins = len(pdb_ids)

    # Put the pdb_ids into the input queue
    for idx, pdb_id in enumerate(pdb_ids):
        input_queue.put((pdb_id, idx))

    # Create a list of processes
    processes = [mp.Process(target=queue_worker, args=(input_queue, output_queue, n_proteins, resolution, method, experiment_folder, verbose)) for _ in range(num_processes)]

    # Start the processes
    for process in processes:
        process.start()

    # Get the results from the output queue
    results = []
    
    for _ in range(n_proteins):
        output, idx = output_queue.get()
        print("Got result", idx + 1, end="\n")
        results.append(output)

    # Wait for all processes to finish
    for idx, process in enumerate(processes):
        process.join()
        print(f"Process {idx + 1} finished\n", end="\n")

    output_file = [OUTPUT_FILE_HEADER]
    output_file += results

    return results

def main():
    args = parse_args()

    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose
    experiment_folder = args.experiment_folder
    experiment_name = args.experiment_name
    sample_uniformly = args.uniform
    protein_subset = args.subset
    num_processes = args.num_processes

    folder_path = os.path.join(experiment_folder, experiment_name)

    # # read the input csv file as a list of (pdb_id, atom_count)
    # with open(input_arg, "r") as f:
    #     all_pdb_ids = list(csv.reader(f))[1:]

    # all_pdb_ids = [(pdb_id, int(atom_count)) for pdb_id, atom_count in all_pdb_ids]

    # if protein_subset == -1:
    #     pdb_ids = all_pdb_ids
    # elif not sample_uniformly:
    #     pdb_ids = random.sample(all_pdb_ids, protein_subset)
    # else:
    #     pdb_ids = pick_uniform_tuples(all_pdb_ids, protein_subset)

    # UNCOMMENT TO VISUALIZE THE ATOM COUNT DISTRIBUTION
    # atom_numbers = [pdb_id[1] for pdb_id in pdb_ids]
    # print(f"Atom numbers: {atom_numbers}")
    # # plot the distribution of atom numbers
    # plt.hist(atom_numbers, bins=30)
    # plt.xlabel("Number of atoms")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of atom numbers")
    # plt.show()
    
    # pdb_ids = [pdb_id[0] for pdb_id in pdb_ids]

    # Read the pdb_ids from the file "protein_sample.txt"
    with open(input_arg, "r") as f:
        pdb_ids = f.readlines()

    print("Start processing")
    output_file = process_protein_batch_in_parallel_sublists(pdb_ids, resolution, method, verbose, folder_path, num_processes)
    print("Finished processing")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    output_filename = os.path.join(folder_path, f"summary.csv")
    try:
        # write the output_file to the output file
        with open(output_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(output_file)
    except:
        print("Error writing the output file\n", end="\n")
        # save it with pickle
        with open(output_filename + ".pickle", "wb") as f:
            pickle.dump(output_file, f)

if __name__ == "__main__":
    main()
