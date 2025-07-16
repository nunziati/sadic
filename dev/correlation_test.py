import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


EXPERIMENT_FOLDER = "/repo/nunziati/sadic/dev/new_experiments"
OUTPUT_FOLDER = "/repo/nunziati/sadic/dev/new_processed_results"
RELOAD = True
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

folder1 = "aligned_spacefill_translated_sphere_0.5"
folder2 = "residuedepth_5_200_all" # "residuedepth_results_survive_5_n_200" # 

folder1 = "single_sadic"
folder2 = "residuedepth_5_200_all_7" # "residuedepth_results_survive_5_n_200" # 

folder1 = "new_subsample_final_30_0.5"
folder2 = "residuedepth_5_200_30_new"

# Helper functions
def load_summary_data(folder, experiment_name):
    """Load summary.csv file for an experiment."""
    file_path = os.path.join(folder, experiment_name, "summary.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_depth_indexes(folder, experiment_name):
    """Load depth indexes from npy files into a single DataFrame."""
    depth_indexes_folder = os.path.join(folder, experiment_name, "depth_indexes")
    if not os.path.exists(depth_indexes_folder):
        raise FileNotFoundError(f"Depth indexes folder not found: {depth_indexes_folder}")
    
    data = []
    for npy_file in tqdm(os.listdir(depth_indexes_folder)):
        if npy_file.endswith(".npy"):
            pdb_code = os.path.splitext(npy_file)[0]
            depth_values = np.load(os.path.join(depth_indexes_folder, npy_file))
            for idx, value in enumerate(depth_values):
                data.append({"pdb_code": pdb_code, "index": idx, "depth": value})
    
    return pd.DataFrame(data)

def load_residuedepth_data(folder):
    tool = "DEPTH"

    if tool == "EDTSurf":
        """Load residue depth data from output-atom.depth files into a single DataFrame."""
        data = []
        for pdb_folder in tqdm(os.listdir(folder)):
            pdb_path = os.path.join(folder, pdb_folder)
            if os.path.isdir(pdb_path):
                depth_file = os.path.join(pdb_path, "output_atom.dep")
                if os.path.exists(depth_file):
                    with open(depth_file, "r") as f:
                        lines = f.readlines()[1:]  # Skip the header line

                    # For each line, spilt by whitespace and remove all the empty strings
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            pdb_code = pdb_folder
                            index = int(parts[0])
                            depth = float(parts[3])
                            data.append({"pdb_code": pdb_code, "index": index, "depth": depth})

        # Create a DataFrame from the collected data
        data = pd.DataFrame(data)
    
    elif tool == "DEPTH":
        """Load residue depth data from output-atom.depth files into a single DataFrame."""
        data = []
        for pdb_folder in tqdm(os.listdir(folder)):
            pdb_path = os.path.join(folder, pdb_folder)
            if os.path.isdir(pdb_path):
                depth_file = os.path.join(pdb_path, "output-atom.depth")
                if os.path.exists(depth_file):
                    df = pd.read_csv(depth_file, sep="\t")
                    df["pdb_code"] = pdb_folder
                    
                    data.append(df)

        data = pd.concat(data, ignore_index=True)

        # Change the name of the column "mean(depth)" to "depth"
        if "mean(depth)" in data.columns:
            data.rename(columns={"mean(depth)": "depth"}, inplace=True)
        else:
            raise ValueError("Column 'mean(depth)' not found in the data.")

    return data

def clean_residuedepth_data(df):
    data_groups = []

    for group, group_df in tqdm(df.groupby("pdb_code")):
        last_index = 0
        last_df_index = None
        for index, row in group_df.iterrows():
            if row["index"] < last_index:
                break
            last_index = row["index"]
            last_df_index = index
        
        if last_df_index is not None:
            data_groups.append(group_df.loc[:last_df_index])
        else:
            data_groups.append(group_df)

    cleaned_df = pd.concat(data_groups, ignore_index=True)

    return cleaned_df

try:
    # Load summary data
    # morphpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, folder1) # "aligned_spacefill_translated_sphere_0.5")
    # inflpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_inflated_translated_sphere_0.5")
    # morphpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_spacefill_reconstructed_sphere_0.5")
    # inflpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_inflated_reconstructed_sphere_0.5")
    # unaligned_summary = load_summary_data(EXPERIMENT_FOLDER, "unaligned_spacefill_translated_sphere_0.5")

    # Load depth indexes
    morphpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, folder1) # "aligned_spacefill_translated_sphere_0.5")
    # inflpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_inflated_translated_sphere_0.5")
    # morphpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_spacefill_reconstructed_sphere_0.5")
    # inflpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_inflated_reconstructed_sphere_0.5")
    # unaligned_depth = load_depth_indexes(EXPERIMENT_FOLDER, "unaligned_spacefill_translated_sphere_0.5")

    # Load residue depth data
    residuedepth_data = load_residuedepth_data(os.path.join(EXPERIMENT_FOLDER, folder2))
except FileNotFoundError as e:
    print(e)
    exit()

print("Clean ResidueDepth data")
residuedepth_data = clean_residuedepth_data(residuedepth_data)

residuedepth_ids = [
    "1d7t", "2a3c", "1hp2", "7ljs", "6c6t", "6svm",
    "3o6x", "6yvd", "3h7l", "4hnt", "4y2l", "4iuu",
    "5thk", "6xo4", "4ysy", "5byv", "7e8b", "1mnq",
    "6p6p", "3dwl", "2j3m", "7rth", "5xon", "2oby",
    "2aaz", "5jz7", "6vk6", "6zfp", "4f49", "5mxd"
]

# residuedepth_ids = ["5thk"]

# filtered_morphpc_translated_depth = morphpc_translated_depth[morphpc_translated_depth["pdb_code"].isin(residuedepth_ids)]
# filtered_residuedepth_data = residuedepth_data[residuedepth_data["pdb_code"].isin(residuedepth_ids)]
filtered_residuedepth_data = residuedepth_data
filtered_morphpc_translated_depth = morphpc_translated_depth

print("Normalize ResidueDepth data")
for pdb_ids in tqdm(filtered_residuedepth_data["pdb_code"].unique()):
    pdb_data = filtered_residuedepth_data[filtered_residuedepth_data["pdb_code"] == pdb_ids]
    if len(pdb_data) > 0:
        min_depth = pdb_data["depth"].min()
        max_depth = pdb_data["depth"].max()
        normalized_depth = 2 * ((pdb_data["depth"]) / (max_depth))
        filtered_residuedepth_data.loc[filtered_residuedepth_data["pdb_code"] == pdb_ids, "depth_normalized"] = normalized_depth

print("Merge data")
all_data = pd.merge(
    filtered_residuedepth_data,
    filtered_morphpc_translated_depth,
    on=["pdb_code", "index"],
    suffixes=("_residuedepth", "_morphpc_translated"),
    how="inner"
)

print("Sort data")
sorted_data = all_data.sort_values(by="depth_normalized").reset_index(drop=True)
sorted_data_first_10_percent = sorted_data.head(int(len(sorted_data) * 0.1))

# Plot scatter plot
print("Plotting scatter plot")
fig, ax = plt.subplots()
ax.scatter(all_data["depth_normalized"],
           all_data["depth_morphpc_translated"], s=1, alpha=0.03)

ax.set_xlabel("Normalized DEPTH")
ax.set_ylabel("SADIC depth index")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "edtsurf_scatter.png"), dpi=300)
plt.close(fig)

print("Scatter plot saved as edtsurf_scatter.png")

# Compute correlation between normalized depth data and depth indices (MorphPC, Translated, Aligned)
pearson_corr, pearson_p = pearsonr(all_data["depth_normalized"], all_data["depth_morphpc_translated"])
spearman_corr, spearman_p = spearmanr(all_data["depth_normalized"], all_data["depth_morphpc_translated"])
print(f"Pearson Correlation (Normalized Depth vs MorphPC Translated Depth): {pearson_corr}, p-value: {pearson_p}")
print(f"Spearman Correlation (Normalized Depth vs MorphPC Translated Depth): {spearman_corr}, p-value: {spearman_p}")

# Compute protein-wise correlations
protein_wise_correlations = []
protein_wise_correlations_dict = {}
for pdb_code, group in tqdm(all_data.groupby("pdb_code")):
    morphpc_depth_values = all_data.loc[
        all_data["pdb_code"] == pdb_code, "depth_morphpc_translated"
    ].values
    pearson_corr, pearson_p = pearsonr(group["depth_normalized"].values, morphpc_depth_values)
    spearman_corr, spearman_p = spearmanr(group["depth_normalized"].values, morphpc_depth_values)
    correlation = {
        "pdb_code": pdb_code,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p
    }
    print(f"Protein {pdb_code}: Pearson Correlation: {pearson_corr}, p-value: {pearson_p}, "
          f"Spearman Correlation: {spearman_corr}, p-value: {spearman_p}")
    # Append the correlation to the list
    protein_wise_correlations.append(correlation)
    protein_wise_correlations_dict[pdb_code] = correlation

# Save the dictionary with pickle
# import pickle
# with open(os.path.join(OUTPUT_FOLDER, "protein_wise_correlations.pkl"), "wb") as f:
#     pickle.dump(protein_wise_correlations_dict, f)="correlation", ignore_index=True)

protein_wise_correlations = pd.DataFrame(protein_wise_correlations)
protein_wise_correlations.to_csv(os.path.join(OUTPUT_FOLDER, "protein_wise_correlations.csv"), index=False)
print("Protein-wise correlations saved to protein_wise_correlations.csv")

# # Compute statistics for protein-wise correlations
# protein_wise_correlation_stats = {
#     "min": protein_wise_correlations.min(),
#     "max": protein_wise_correlations.max(),
#     "avg": protein_wise_correlations.mean(),
#     "std": protein_wise_correlations.std(),
# }

# print("Protein-wise Correlation Stats (Normalized Depth vs MorphPC Translated Depth):")
# print(f"Min: {protein_wise_correlation_stats['min']}")
# print(f"Max: {protein_wise_correlation_stats['max']}")
# print(f"Avg: {protein_wise_correlation_stats['avg']}")
# print(f"Std: {protein_wise_correlation_stats['std']}")

# Plot histogram of protein-wise correlations
fig, ax = plt.subplots()
ax.hist(protein_wise_correlations["pearson_corr"], bins=50, edgecolor="black")
ax.set_title("Protein-wise Correlations (Normalized Depth vs MorphPC Translated Depth)")
ax.set_xlabel("Correlation Coefficient")
ax.set_ylabel("Frequency")
fig.savefig(os.path.join(OUTPUT_FOLDER, "protein_wise_correlations_histogram.pdf"), dpi=300)