import argparse
import csv
import random
import multiprocessing as mp
import queue
import pickle

import matplotlib.pyplot as plt
import numpy as np

from process_single_protein import process_protein

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DEFAULT_INPUT = "proteins_dataset_PDB_with_protein_size.csv"
DEFAULT_OUTUPT = "output2_translated_sphere.csv"
DEFAULT_RESOLUTION = 0.5
DEFAULT_METHOD = "translated_sphere_vectorized"
DEFAULT_VERBOSE = False
DEFAULT_SUBSET = 100
DEFAULT_UNIFORM = True
DEFAULT_RESUME = -1
DEFAULT_NUM_PROCESSES = 8

OUTPUT_FILE_HEADER = ("PDB_ID", "resolution", "method", "t1", "t2", "t3", "t4", "N", "n", "n_1", "n_1_raw_protein_int_volume", "n_2", "n_2_components", "n_2_filled_voxels", "n_2_filled_protein_int_volume", "n_3", "n_4", "p_4_min", "p_4_max", "p_4_avg", "p_4_med", "p_4_std")

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTUPT, help="Output CSV file")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET, help="Number of proteins to process")
    parser.add_argument("--uniform", action="store_true", default=DEFAULT_UNIFORM, help="Sample uniformly on the number of atoms")
    parser.add_argument("--resume", type=int, default=DEFAULT_RESUME, help="Index of the protein to resume from")
    parser.add_argument("--num_processes", type=int, default=DEFAULT_NUM_PROCESSES, help="Number of processes to use for parallel processing")
    return parser.parse_args()

def process_single_protein_and_extract_output(pdb_id, resolution=0.3, method=None, verbose=True):
    try:
        output = process_protein(pdb_id.strip(), resolution=resolution, method=method, verbose=verbose)
    except KeyboardInterrupt as e:
        print("Interrupted by the user\n", end="\n")
        raise e
    except:
        print(f"Error processing protein {pdb_id.strip()}\n", end="\n")
        return tuple([pdb_id.strip(), "ERROR"] + [None] * 19)

    output_tuple = (
        pdb_id.strip(),
        resolution,
        method,
        output["times"][0],
        output["times"][1],
        output["times"][2],
        output["times"][3],
        output["complexity_variables"]["N"],
        output["complexity_variables"]["n"],
        output["complexity_variables"]["1"]["n"] if "n" in output["complexity_variables"]["1"] else None,
        output["complexity_variables"]["1"]["raw_protein_int_volume"] if "raw_protein_int_volume" in output["complexity_variables"]["1"] else None,
        output["complexity_variables"]["2"]["n"] if "n" in output["complexity_variables"]["2"] else None,
        output["complexity_variables"]["2"]["n_components"] if "n_components" in output["complexity_variables"]["2"] else None,
        output["complexity_variables"]["2"]["n_filled_voxels"] if "n_filled_voxels" in output["complexity_variables"]["2"] else None,
        output["complexity_variables"]["2"]["n_protein_int_volume"] if "n_protein_int_volume" in output["complexity_variables"]["2"] else None,
        output["complexity_variables"]["3"]["n"] if "n" in output["complexity_variables"]["3"] else None,
        output["complexity_variables"]["4"]["n"] if "n" in output["complexity_variables"]["4"] else None,
        np.array(output["complexity_variables"]["4"]["p_list"]).min() if "p_list" in output["complexity_variables"]["4"] else None,
        np.array(output["complexity_variables"]["4"]["p_list"]).max() if "p_list" in output["complexity_variables"]["4"] else None,
        np.array(output["complexity_variables"]["4"]["p_list"]).mean() if "p_list" in output["complexity_variables"]["4"] else None,
        np.median(np.array(output["complexity_variables"]["4"]["p_list"])) if "p_list" in output["complexity_variables"]["4"] else None,
        np.array(output["complexity_variables"]["4"]["p_list"]).std() if "p_list" in output["complexity_variables"]["4"] else None,
    )
    
    return output_tuple

def process_protein_batch(pdb_ids, resolution=0.3, method=None, verbose=True):
    # prepare the output_file as a list of tuples
    output_file = []
    
    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, verbose=verbose)

        output_file.append(output_tuple)

    return output_file

def pick_uniform_tuples(tuples_list, N):
    # Sort the tuples by the second element
    sorted_tuples = sorted(tuples_list, key=lambda x: x[1])

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

def sublist_worker(sublist, resolution, method, verbose):
    return process_protein_batch(sublist, resolution, method, verbose)

def queue_worker(input_queue, output_queue, n_proteins, resolution, method, verbose):
    while not input_queue.empty():
        try:
            pdb_id, idx = input_queue.get_nowait()
        except queue.Empty as e:
            print(e)
            break
        except Exception as e:
            raise e

        print(f"Processing protein {idx + 1}/{n_proteins}\n", end="\n")
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, verbose=verbose)
        output_queue.put(output_tuple)

def process_protein_batch_scalar(pdb_ids, resolution, method, verbose):
    output_file = [OUTPUT_FILE_HEADER]

    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        
        output_tuple = process_single_protein_and_extract_output(pdb_id, resolution=resolution, method=method, verbose=verbose)

        output_file.append(output_tuple)

    return output_file


def process_protein_batch_in_parallel_sublists(pdb_ids, resolution, method, verbose, num_processes):
    # Split the pdb_ids into sublists for each process
    sublists = [pdb_ids[i::num_processes] for i in range(num_processes)]
    
    # Create a multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the worker function
        results = pool.starmap(sublist_worker, [(sublist, resolution, method, verbose) for sublist in sublists])
    
    # Merge the results from all processes
    merged_results = [item for sublist in results for item in sublist[1:]]
    
    output_file = [OUTPUT_FILE_HEADER]
    output_file += merged_results

    return merged_results

def process_protein_batch_in_parallel_queue(pdb_ids, resolution, method, verbose, num_processes):
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    n_proteins = len(pdb_ids)

    # Put the pdb_ids into the input queue
    for idx, pdb_id in enumerate(pdb_ids):
        input_queue.put((pdb_id, idx))

    # Create a list of processes
    processes = [mp.Process(target=queue_worker, args=(input_queue, output_queue, n_proteins, resolution, method, verbose)) for _ in range(num_processes)]

    # Start the processes
    for process in processes:
        process.start()

    # Get the results from the output queue
    results = []
    
    for _ in range(n_proteins):
        results.append(output_queue.get())

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
    output_filename = args.output
    sample_uniformly = args.uniform
    protein_subset = args.subset
    num_processes = args.num_processes

    # read the input csv file as a list of (pdb_id, atom_count)
    with open(input_arg, "r") as f:
        all_pdb_ids = list(csv.reader(f))[1:]

    all_pdb_ids = [(pdb_id, int(atom_count)) for pdb_id, atom_count in all_pdb_ids]

    if protein_subset == -1:
        pdb_ids = all_pdb_ids
    elif not sample_uniformly:
        pdb_ids = random.sample(all_pdb_ids, protein_subset)
    else:
        pdb_ids = pick_uniform_tuples(all_pdb_ids, protein_subset)

    # UNCOMMENT TO VISUALIZE THE ATOM COUNT DISTRIBUTION
    # atom_numbers = [pdb_id[1] for pdb_id in pdb_ids]
    # print(f"Atom numbers: {atom_numbers}")
    # # plot the distribution of atom numbers
    # plt.hist(atom_numbers, bins=30)
    # plt.xlabel("Number of atoms")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of atom numbers")
    # plt.show()
    
    pdb_ids = [pdb_id[0] for pdb_id in pdb_ids]

    print("Start processing")
    output_file = process_protein_batch_in_parallel_queue(pdb_ids, resolution, method, verbose, num_processes)
    print("Finished processing")

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