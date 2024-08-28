import csv
import numpy as np


input_arg = "proteins_dataset_PDB_with_protein_size.csv"
protein_subset = 1000

def pick_uniform_tuples(tuples_list, N):
    elements_dict = {x: y for x, y in tuples_list}

    reconstructed_tuples = [(x, y) for x, y in elements_dict.items()]

    # Sort the tuples by the second element
    sorted_tuples = sorted(reconstructed_tuples, key=lambda x: x[1])

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
        second_elements[idx] = -100000000
    
    # Select the tuples corresponding to these indices
    selected_tuples = [sorted_tuples[i] for i in indices]
    
    return selected_tuples

# read the input csv file as a list of (pdb_id, atom_count)
with open(input_arg, "r") as f:
    all_pdb_ids = list(csv.reader(f))[1:]

all_pdb_ids = [(pdb_id, int(atom_count)) for pdb_id, atom_count in all_pdb_ids]

pdb_ids = pick_uniform_tuples(all_pdb_ids, protein_subset)

pdb_ids = [pdb_id[0] for pdb_id in pdb_ids]

# Save the extracted pdb_ids to a new txt file called "protein_sample.txt"
with open("protein_sample.txt", "w") as f:
    for pdb_id in pdb_ids:
        f.write(f"{pdb_id}\n")
