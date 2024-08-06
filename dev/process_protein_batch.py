import argparse
import csv
import random

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
    return parser.parse_args()

def process_protein_batch(pdb_ids, resolution=0.3, method=None, verbose=True):
    # prepare the output_file as a list of tuples
    output_file = [("PDB_ID", "resolution", "method", "t1", "t2", "t3", "t4", "N", "n", "n_1", "p_1_min", "p_1_max", "p_1_avg", "p_1_med", "p_1_std", "n_2", "p_2_min", "p_2_max", "p_2_avg", "p_2_med", "p_2_std", "n_3", "p_3_min", "p_3_max", "p_3_avg", "p_3_med", "p_3_std", "n_4", "p_4_min", "p_4_max", "p_4_avg", "p_4_med", "p_4_std")]
    
    for idx, pdb_id in enumerate(pdb_ids):
        print(f"Processing protein {idx + 1}/{len(pdb_ids)}\n", end="\n")
        output = process_protein(pdb_id.strip(), resolution=resolution, method=method, verbose=verbose)

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
            np.array(output["complexity_variables"]["1"]["p_list"]).min() if "p_list" in output["complexity_variables"]["1"] else None,
            np.array(output["complexity_variables"]["1"]["p_list"]).max() if "p_list" in output["complexity_variables"]["1"] else None,
            np.array(output["complexity_variables"]["1"]["p_list"]).mean() if "p_list" in output["complexity_variables"]["1"] else None,
            np.median(np.array(output["complexity_variables"]["1"]["p_list"])) if "p_list" in output["complexity_variables"]["1"] else None,
            np.array(output["complexity_variables"]["1"]["p_list"]).std() if "p_list" in output["complexity_variables"]["1"] else None,
            output["complexity_variables"]["2"]["n"] if "n" in output["complexity_variables"]["2"] else None,
            np.array(output["complexity_variables"]["2"]["p_list"]).min() if "p_list" in output["complexity_variables"]["2"] else None,
            np.array(output["complexity_variables"]["2"]["p_list"]).max() if "p_list" in output["complexity_variables"]["2"] else None,
            np.array(output["complexity_variables"]["2"]["p_list"]).mean() if "p_list" in output["complexity_variables"]["2"] else None,
            np.median(np.array(output["complexity_variables"]["2"]["p_list"])) if "p_list" in output["complexity_variables"]["2"] else None,
            np.array(output["complexity_variables"]["2"]["p_list"]).std() if "p_list" in output["complexity_variables"]["2"] else None,
            output["complexity_variables"]["3"]["n"] if "n" in output["complexity_variables"]["3"] else None,
            np.array(output["complexity_variables"]["3"]["p_list"]).min() if "p_list" in output["complexity_variables"]["3"] else None,
            np.array(output["complexity_variables"]["3"]["p_list"]).max() if "p_list" in output["complexity_variables"]["3"] else None,
            np.array(output["complexity_variables"]["3"]["p_list"]).mean() if "p_list" in output["complexity_variables"]["3"] else None,
            np.median(np.array(output["complexity_variables"]["3"]["p_list"])) if "p_list" in output["complexity_variables"]["3"] else None,
            np.array(output["complexity_variables"]["3"]["p_list"]).std() if "p_list" in output["complexity_variables"]["3"] else None,
            output["complexity_variables"]["4"]["n"] if "n" in output["complexity_variables"]["4"] else None,
            np.array(output["complexity_variables"]["4"]["p_list"]).min() if "p_list" in output["complexity_variables"]["4"] else None,
            np.array(output["complexity_variables"]["4"]["p_list"]).max() if "p_list" in output["complexity_variables"]["4"] else None,
            np.array(output["complexity_variables"]["4"]["p_list"]).mean() if "p_list" in output["complexity_variables"]["4"] else None,
            np.median(np.array(output["complexity_variables"]["4"]["p_list"])) if "p_list" in output["complexity_variables"]["4"] else None,
            np.array(output["complexity_variables"]["4"]["p_list"]).std() if "p_list" in output["complexity_variables"]["4"] else None,
        )

        output_file.append(output_tuple)

    return output_file

def main():
    args = parse_args()

    input_arg = args.input
    resolution = args.resolution
    method = args.method
    verbose = args.verbose
    output_filename = args.output
    protein_subset = args.subset

    # read the input file as a list of PDB IDs
    with open(input_arg, "r") as f:
        all_pdb_ids = f.readlines()

    pdb_ids = all_pdb_ids if protein_subset == -1 else random.sample(all_pdb_ids, protein_subset)

    output_file = process_protein_batch(pdb_ids, resolution, method, verbose)

    # write the output_file to the output file
    with open(output_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_file)

if __name__ == "__main__":
    main()