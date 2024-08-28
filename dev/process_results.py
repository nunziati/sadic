import pandas as pd
import numpy as np
from tqdm import tqdm
import os

experiment_folder = "experiments/aligned_spacefill_translated_sphere"
input_name = os.path.join(experiment_folder, "summary.csv")
output_name = os.path.join(experiment_folder, "summary_processed.csv")

df = pd.read_csv(input_name)

df["N*p_idx_max/n"] = df["N"] * df["p_idx_max"] / df["n"]
df["N*p_idx_avg/n"] = df["N"] * df["p_idx_avg"] / df["n"]
df["idx_visited_voxels"] = 0
df["max_idx_visited_voxel_map"] = 0
df["min_idx_visited_voxel_map"] = 0
df["avg_idx_visited_voxel_map"] = 0.
df["std_idx_visited_voxel_map"] = 0.

# Iterate the rows of df
for i, row in tqdm(df.iterrows()):
    pdb_id = row["PDB_ID"]
    path = os.path.join(experiment_folder, "indexes_computation", "voxel_operations_map", f"{pdb_id}.npy")
    if not os.path.exists(path):
        print("File not found:", path)
        continue
    
    voxel_map = np.load(path)
    voxel_centers = np.argwhere(voxel_map)
    values = voxel_map[voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]]
    df.at[i, "idx_visited_voxels"] = len(values)
    df.at[i, "max_idx_visited_voxel_map"] = values.max()
    df.at[i, "min_idx_visited_voxel_map"] = values.min()
    df.at[i, "avg_idx_visited_voxel_map"] = values.mean()
    df.at[i, "std_idx_visited_voxel_map"] = values.std()

df.to_csv(output_name, index=False)