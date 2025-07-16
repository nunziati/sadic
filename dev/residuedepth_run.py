import os
import urllib.request
import subprocess
from tqdm import tqdm

def download_pdb(pdb_id, output_dir="."):
    """Download a PDB file from the RCSB PDB website."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    try:
        urllib.request.urlretrieve(url, pdb_file)
        print(f"Downloaded {pdb_id}.pdb")
        return pdb_file
    except Exception as e:
        print(f"Failed to download {pdb_id}.pdb: {e}")
        return None

def process_proteins(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # shuffle the lines to process proteins in random order
    import random
    random.shuffle(lines)

    # lines = ["1ubq", "1ema", "1hrc", "1lyz", "1mbn", "2ptc", "4ins"]
    count = 0

    for line in tqdm(lines):
        pdb_id = line.strip()
        pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
        
        # Download the PDB file if it doesn't exist
        if not os.path.exists(pdb_file):
            print(f"File {pdb_file} not found. Attempting to download...")
            pdb_file = download_pdb(pdb_id, output_dir)
            if not pdb_file:
                print(f"Skipping {pdb_id} due to download failure.")
                continue
        
        try:
            if os.path.join(output_dir, pdb_id) in os.listdir(output_dir):
                print(f"Directory for {pdb_id} already exists. Skipping...")
                continue

            print(f"Processing {pdb_id}...")
            # Create the output directory for the current PDB ID
            pdb_output_dir = os.path.join(output_dir, pdb_id)
            os.makedirs(pdb_output_dir, exist_ok=True)

            # Construct the EDTSurf command
            depth_command = f"/repo/nunziati/sadic/dev/EDTSurf/EDTSurf -i {pdb_file} -o {pdb_output_dir}/output -s 0"
            depth_command = f"/repo/nunziati/depth/depth_source/bin/DEPTH -i {pdb_file} -o {pdb_output_dir}/output -survive 5 -n 200"
            print(f"Running EDTSurf for {pdb_id}...")
            
            # Execute the EDTSurf command
            subprocess.run(depth_command, shell=True, check=True)
            print(f"EDTSurf processing complete for {pdb_id}. Results saved to {pdb_output_dir}/output")
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} proteins so far.")
        except FileNotFoundError:
            print(f"EDTSurf executable not found.")
        except subprocess.CalledProcessError as e:
            print(f"Error running EDTSurf for {pdb_id}: {e}")
        finally:
            # Remove the PDB file after processing
            if os.path.exists(pdb_file):
                os.remove(pdb_file)
                print(f"Deleted {pdb_file}")

if __name__ == "__main__":
    input_file = "/repo/nunziati/sadic/dev/protein_sample_40000_subsample_new.txt"
    output_dir = "/repo/nunziati/sadic/dev/new_experiments/residuedepth_5_200_30_new"
    process_proteins(input_file, output_dir)
    print(f"Processing complete. EDTSurf results saved to {output_dir}")