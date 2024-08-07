import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np

from process_single_protein import process_protein

random.seed(42)

DEFAULT_INPUT = "input.txt"
DEFAULT_OUTUPT = "output.csv"
DEFAULT_RESOLUTION = 0.3
DEFAULT_METHOD = "basic_vectorized"
DEFAULT_VERBOSE = False
DEFAULT_SUBSET = -1

def parse_args():
    parser = argparse.ArgumentParser(description="SADIC: Solvent Accessible Depth Index Calculator")
    parser.add_argument("input", type=str, default=DEFAULT_INPUT, help="Input PDB file or PDB ID")
    parser.add_argument("output", type=str, default=DEFAULT_OUTUPT, help="Output CSV file")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="Resolution of the grid")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, help="Method to use for the algorithm")
    parser.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Prints the progress of the algorithm")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET, help="Number of proteins to process")
    parser.add_argument("--uniform", action="store_true", help="Sample uniformly on the number of atoms")
    parser.add_argument("--resume", type=int, default=-1, help="Index of the protein to resume from")
    return parser.parse_args()

def process_protein_batch(pdb_ids, resolution=0.3, method=None, verbose=True):
    # prepare the output_file as a list of tuples
    output_file = [("PDB_ID", "resolution", "method", "t1", "t2", "t3", "t4", "N", "n", "n_1", "n_1_raw_protein_int_volume", "n_2", "n_2_components", "n_2_filled_voxels", "n_2_filled_protein_int_volume", "n_3", "n_4", "p_4_min", "p_4_max", "p_4_avg", "p_4_med", "p_4_std")]
    
    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        
        try:
            output = process_protein(pdb_id.strip(), resolution=resolution, method=method, verbose=verbose)
        except:
            print(f"Error processing protein {pdb_id.strip()}\n", end="\n")
            output_tuple = tuple([pdb_id.strip(), "ERROR"] + [None] * (len(output_file[0]) - 2))
            output_file.append(output_tuple)
            continue

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

        output_file.append(output_tuple)

    return output_file

def pick_uniform_tuples(tuples_list, N):
    # Sort the tuples by the second element
    sorted_tuples = sorted(tuples_list, key=lambda x: x[1])
    
    # Extract the second elements and calculate the empirical CDF
    second_elements = [y for _, y in sorted_tuples]
    cdf_values = np.linspace(0, 1, len(second_elements), endpoint=False) + (0.5 / len(second_elements))
    
    # Generate N random numbers uniformly distributed between 0 and 1
    random_numbers = sorted(random.random() for _ in range(N))
    
    # Use the empirical CDF to map random numbers to indices
    indices = np.searchsorted(cdf_values, random_numbers)
    
    # Select the tuples corresponding to these indices
    selected_tuples = [sorted_tuples[i] for i in indices]
    
    return selected_tuples

def main():
    args = parse_args()

    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose
    output_filename = args.output
    sample_uniformly = args.uniform
    protein_subset = args.subset

    # read the input file as a list of PDB IDs
    with open(input_arg, "r") as f:
        all_pdb_ids = f.readlines()

    if protein_subset == -1:
        pdb_ids = all_pdb_ids
    elif not sample_uniformly:
        pdb_ids = random.sample(all_pdb_ids, protein_subset)
    else:
        pdb_ids = pick_uniform_tuples(all_pdb_ids, protein_subset)

    atom_numbers = [pdb_id[1] for pdb_id in pdb_ids]
    pdb_ids = [pdb_id[0] for pdb_id in pdb_ids]

    # plot the distribution of atom numbers
    plt.hist(atom_numbers, bins=100)
    plt.xlabel("Number of atoms")
    plt.ylabel("Frequency")
    plt.title("Distribution of atom numbers")
    plt.show()
    

    output_file = process_protein_batch(pdb_ids, resolution, method, verbose)

    # write the output_file to the output file
    with open(output_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_file)

if __name__ == "__main__":
    main()