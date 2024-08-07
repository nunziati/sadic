import csv
from tqdm import tqdm

import multiprocessing as mp

from sadic.pdb import PDBEntity

# Function to count atoms in a PDB structure
def count_atoms(pdb_id):
    protein = PDBEntity(pdb_id)
    model = protein.models[1]
    atoms = model.get_centers()
    atom_count = atoms.shape[0]

    return pdb_id, atom_count

# Worker function to process a sublist of PDB IDs
def process_pdb_ids(pdb_ids_sublist):
    pdb_atom_counts_sublist = {}
    for pdb_id in pdb_ids_sublist:
        try:
            pdb_id, atom_count = count_atoms(pdb_id)
            pdb_atom_counts_sublist[pdb_id] = atom_count
            print("Processed", pdb_id, "with", atom_count, "atoms")
        except Exception as e:
            print(f"Failed to process {pdb_id}: {e}")
    return pdb_atom_counts_sublist

def main(input_file, output_file, num_processes):
    # Read the PDB IDs from the input file
    with open(input_file, "r") as f:
        all_pdb_ids = f.read().splitlines()

    # Divide the PDB IDs into sublists for each process
    sublists = [all_pdb_ids[i::num_processes] for i in range(num_processes)]

    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_pdb_ids, sublists)

    # Merge the results from all processes
    pdb_atom_counts = {}
    for result in results:
        pdb_atom_counts.update(result)

    pdb_ids_with_protein_size = [("PDB ID", "N")]
    
    for pdb_id, atom_count in pdb_atom_counts.items():
        pdb_ids_with_protein_size.append((pdb_id, atom_count))

    # Write the results to the output file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(pdb_ids_with_protein_size)

if __name__ == '__main__':
    input_file = "proteins_dataset_PDB.txt"
    output_file = "proteins_dataset_PDB_with_protein_size.csv"
    num_processes = 100  # Customize the number of processes
    main(input_file, output_file, num_processes)