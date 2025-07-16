import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from scipy.stats import spearmanr

# Define paths
EXPERIMENT_FOLDER = "/repo/nunziati/sadic/dev/new_experiments"
OUTPUT_FOLDER = "/repo/nunziati/sadic/dev/new_processed_results"
RELOAD = True
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

report = []

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

    for group, group_df in df.groupby("pdb_code"):
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


def calculate_metrics(data1, data2):
    """Calculate MSD, MAD, PW-MSD, and PW-MAD."""
    di1 = data1["depth"]
    di2 = data2["depth"]
    differences = di1 - di2
    msd = np.mean(differences)
    mad = np.mean(np.abs(differences))
    pw_sd = differences.groupby(morphpc_translated_depth["pdb_code"]).mean()
    pw_ad = differences.abs().groupby(morphpc_translated_depth["pdb_code"]).mean()
    pw_msd = np.mean(pw_sd)
    pw_mad = np.mean(pw_ad)
    return msd, mad, pw_msd, pw_mad

def save_plot(fig, filename):
    """Save a matplotlib figure."""
    fig.savefig(os.path.join(OUTPUT_FOLDER, filename))
    plt.close(fig)

def save_report(report, filename):
    """Save a text report."""
    with open(os.path.join(OUTPUT_FOLDER, filename), "w") as f:
        f.write("\n".join(report))

def save_data(morphpc_translated_summary, inflpc_translated_summary, morphpc_reconstructed_summary,
              inflpc_reconstructed_summary, unaligned_summary, morphpc_translated_depth,
              inflpc_translated_depth, morphpc_reconstructed_depth, inflpc_reconstructed_depth,
              unaligned_depth, residuedepth_data):
    """Save data to a pickle file."""

    data = {
        "morphpc_translated_summary": morphpc_translated_summary,
        "inflpc_translated_summary": inflpc_translated_summary,
        "morphpc_reconstructed_summary": morphpc_reconstructed_summary,
        "inflpc_reconstructed_summary": inflpc_reconstructed_summary,
        "unaligned_summary": unaligned_summary,
        "morphpc_translated_depth": morphpc_translated_depth,
        "inflpc_translated_depth": inflpc_translated_depth,
        "morphpc_reconstructed_depth": morphpc_reconstructed_depth,
        "inflpc_reconstructed_depth": inflpc_reconstructed_depth,
        "unaligned_depth": unaligned_depth,
        "residuedepth_data": residuedepth_data,
    }

    # Save the dataframe to pickle files
    for name, df in data.items():
        pickle_path = os.path.join(OUTPUT_FOLDER, f"{name}.pkl")
        df.to_pickle(pickle_path)
        print(f"Saved {name} to {pickle_path}")

def plot_histograms(metrics, labels, folder_name, binwidth=None, viewbins=None, **kwargs):
    """Plot histograms for given metrics and save them in a folder."""
    folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    complete_names = {
        "MSD": "Mean Signed Deviation",
        "MAD": "Mean Absolute Deviation",
        "PW-MSD": "Protein-Wise Mean Squared Deviation",
        "PW-MAD": "Protein-Wise Mean Absolute Deviation"
    }

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        fig, ax = plt.subplots()

        # Use the specified binwidth for histogram
        if binwidth is not None:
            bins = np.arange(metric.min(), metric.max() + binwidth, binwidth)
        elif viewbins is not None and "xlim" in kwargs:
            bins = np.linspace(kwargs["xlim"][i][0], kwargs["xlim"][i][1], viewbins)
        else:
            bins = 30

        ax.hist(metric, bins=bins, edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")

        ax.yaxis.offsetText.set_fontsize(16)
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        ax.tick_params(axis='both', labelsize=16)
        fig.tight_layout()

        # Set x-axis limits if provided
        if "xlim" in kwargs and len(kwargs["xlim"]) == len(metrics):
            ax.set_xlim(kwargs["xlim"][i])
        
        save_plot(fig, os.path.join(folder_path, f"{label}.pdf"))

if RELOAD:
    # Load data
    try:
        # Load summary data
        morphpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_spacefill_translated_sphere_0.5") # "aligned_spacefill_translated_sphere_0.5")
        inflpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_inflated_translated_sphere_0.5")
        morphpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_spacefill_reconstructed_sphere_0.5")
        inflpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_inflated_reconstructed_sphere_0.5")
        unaligned_summary = load_summary_data(EXPERIMENT_FOLDER, "unaligned_spacefill_translated_sphere_0.5")
        
        # Load depth indexes
        morphpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_spacefill_translated_sphere_0.5") # "aligned_spacefill_translated_sphere_0.5")
        inflpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_inflated_translated_sphere_0.5")
        morphpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_spacefill_reconstructed_sphere_0.5")
        inflpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_inflated_reconstructed_sphere_0.5")
        unaligned_depth = load_depth_indexes(EXPERIMENT_FOLDER, "unaligned_spacefill_translated_sphere_0.5")
        
        # morphpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_spacefill_translated_sphere_0.5")
        # inflpc_translated_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_translated_sphere_0.5")
        # morphpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_spacefill_0.5")
        # inflpc_reconstructed_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_0.5")
        # unaligned_summary = load_summary_data(EXPERIMENT_FOLDER, "aligned_0.5")

        # # Load depth indexes
        # morphpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_spacefill_translated_sphere_0.5")
        # inflpc_translated_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_translated_sphere_0.5")
        # morphpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_spacefill_0.5")
        # inflpc_reconstructed_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_0.5")
        # unaligned_depth = load_depth_indexes(EXPERIMENT_FOLDER, "aligned_0.5")

        # morphpc_translated_summary = morphpc_translated_summary[morphpc_translated_summary["resolution"] != "ERROR"]
        # inflpc_translated_summary = inflpc_translated_summary[inflpc_translated_summary["resolution"] != "ERROR"]
        # morphpc_reconstructed_summary = morphpc_reconstructed_summary[morphpc_reconstructed_summary["resolution"] != "ERROR"]
        # inflpc_reconstructed_summary = inflpc_reconstructed_summary[inflpc_reconstructed_summary["resolution"] != "ERROR"]
        # unaligned_summary = unaligned_summary[unaligned_summary["resolution"] != "ERROR"]

        # Load residue depth data
        residuedepth_data = load_residuedepth_data(os.path.join(EXPERIMENT_FOLDER, "residuedepth_results_survive_5_n_200"))
    except FileNotFoundError as e:
        print(e)
        exit()

    # Save data to pickle files
    save_data(morphpc_translated_summary, inflpc_translated_summary, morphpc_reconstructed_summary,
              inflpc_reconstructed_summary, unaligned_summary, morphpc_translated_depth,
              inflpc_translated_depth, morphpc_reconstructed_depth, inflpc_reconstructed_depth,
              unaligned_depth, residuedepth_data)
    print("Data loaded and saved to pickle files.")

else:
    # Load data from pickle files
    morphpc_translated_summary = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "morphpc_translated_summary.pkl"))
    inflpc_translated_summary = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "inflpc_translated_summary.pkl"))
    morphpc_reconstructed_summary = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "morphpc_reconstructed_summary.pkl"))
    inflpc_reconstructed_summary = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "inflpc_reconstructed_summary.pkl"))
    unaligned_summary = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "unaligned_summary.pkl"))
    
    morphpc_translated_depth = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "morphpc_translated_depth.pkl"))
    inflpc_translated_depth = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "inflpc_translated_depth.pkl"))
    morphpc_reconstructed_depth = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "morphpc_reconstructed_depth.pkl"))
    inflpc_reconstructed_depth = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "inflpc_reconstructed_depth.pkl"))
    unaligned_depth = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "unaligned_depth.pkl"))
    
    # morphpc_translated_summary = morphpc_translated_summary[morphpc_translated_summary["resolution"] != "ERROR"]
    # inflpc_translated_summary = inflpc_translated_summary[inflpc_translated_summary["resolution"] != "ERROR"]
    # morphpc_reconstructed_summary = morphpc_reconstructed_summary[morphpc_reconstructed_summary["resolution"] != "ERROR"]
    # inflpc_reconstructed_summary = inflpc_reconstructed_summary[inflpc_reconstructed_summary["resolution"] != "ERROR"]
    # unaligned_summary = unaligned_summary[unaligned_summary["resolution"] != "ERROR"]

    residuedepth_data = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "residuedepth_data.pkl"))

# Clean ResidueDepth data
residuedepth_data = clean_residuedepth_data(residuedepth_data)

# Example: Print loaded data shapes
print("MorphPC Translated Summary Shape:", morphpc_translated_summary.shape)
print("MorphPC Translated Depth Shape:", morphpc_translated_depth.shape)
print("Residue Depth Data Shape:", residuedepth_data.shape)

# 1. Bounding box analysis
unaligned_N = unaligned_summary["N"]
unaligned_n = unaligned_summary["n"]
aligned_N = morphpc_translated_summary["N"]
aligned_n = morphpc_translated_summary["n"]

# Scatter plot
fig, ax = plt.subplots()
ax.scatter(unaligned_N, unaligned_n, color="tab:blue", label="Unaligned", s=2)
ax.scatter(aligned_N, aligned_n, color="tab:red", label="Aligned", s=2)
ax.set_xlabel("N (Number of Atoms)")
ax.set_ylabel("n (Number of Voxels)")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_xticks([0, 1e4, 2e4, 3e4, 4e4])
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=11) #, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2)  # Place legend above plot, outside
fig.tight_layout()
save_plot(fig, "bounding_box_scatter.pdf")

# Percentage volume reduction
print("Volume reduction")
volume_reduction = np.maximum((unaligned_n - aligned_n) / unaligned_n * 100, 0)
volume_reduction_stats = {
    "min": volume_reduction.min(),
    "max": volume_reduction.max(),
    "avg": volume_reduction.mean(),
    "std": volume_reduction.std(),
}

report.append("Bounding Box Analysis:")
report.append(f"Volume Reduction Stats: {volume_reduction_stats}")
report.append("")

# Histogram
fig, ax = plt.subplots()
ax.hist(volume_reduction, bins=30, edgecolor="black")
ax.set_xlabel("Volume Reduction (%)")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "volume_reduction_histogram.pdf")

# 2 & 3. Reconstructed vs translated sphere metrics
print("Translated sphere (MorphPC)")
# MorphPC
morphpc_translated_di = morphpc_translated_depth["depth"]
morphpc_reconstructed_di = morphpc_reconstructed_depth["depth"]

morphpc_reconstructed_translated_msd, morphpc_reconstructed_translated_mad, morphpc_reconstructed_translated_pw_msd, morphpc_reconstructed_translated_pw_mad = calculate_metrics(
    morphpc_translated_depth, morphpc_reconstructed_depth
)

report.append("Reconstructed vs Translated Sphere Metrics (MorphPC):")
report.append(f"MSD: {morphpc_reconstructed_translated_msd}")
report.append(f"MAD: {morphpc_reconstructed_translated_mad}")
report.append(f"PW-MSD: {morphpc_reconstructed_translated_pw_msd}")
report.append(f"PW-MAD: {morphpc_reconstructed_translated_pw_mad}")
report.append("")

# Calculate metrics for MorphPC
morphpc_sd = (morphpc_reconstructed_di - morphpc_translated_di)
morphpc_ad = pd.Series(np.abs(morphpc_reconstructed_di - morphpc_translated_di))
morphpc_pw_sd = morphpc_sd.groupby(morphpc_translated_depth["pdb_code"]).mean()
morphpc_pw_ad = morphpc_ad.groupby(morphpc_translated_depth["pdb_code"]).mean()

# Plot histograms for MorphPC
plot_histograms(
    [morphpc_sd, morphpc_ad, morphpc_pw_sd, morphpc_pw_ad],
    ["MSD", "MAD", "PW-MSD", "PW-MAD"],
    "morphpc_using_translated_sphere_metrics_histograms",
    viewbins=50,
    xlim=[(-0.05, 0.05), (-0.05, 0.05), (-0.005, 0.005), (-0.05, 0.05)]
)

# InflPC
print("Translated sphere (InflPC)")
inflpc_translated_di = inflpc_translated_depth["depth"]
inflpc_reconstructed_di = inflpc_reconstructed_depth["depth"]

inflpc_reconstructed_translated_msd, inflpc_reconstructed_translated_mad, inflpc_reconstructed_translated_pw_msd, inflpc_reconstructed_translated_pw_mad = calculate_metrics(
    inflpc_translated_depth, inflpc_reconstructed_depth
)
report.append("Reconstructed vs Translated Sphere Metrics (InflPC):")
report.append(f"MSD: {inflpc_reconstructed_translated_msd}")
report.append(f"MAD: {inflpc_reconstructed_translated_mad}")
report.append(f"PW-MSD: {inflpc_reconstructed_translated_pw_msd}")
report.append(f"PW-MAD: {inflpc_reconstructed_translated_pw_mad}")
report.append("")

# Calculate metrics for InflPC
inflpc_sd = (inflpc_reconstructed_di - inflpc_translated_di)
inflpc_ad = pd.Series(np.abs(inflpc_reconstructed_di - inflpc_translated_di))
inflpc_pw_sd = inflpc_sd.groupby(inflpc_translated_depth["pdb_code"]).mean()
inflpc_pw_ad = inflpc_ad.groupby(inflpc_translated_depth["pdb_code"]).mean()

# Plot histograms for InflPC
plot_histograms(
    [inflpc_sd, inflpc_ad, inflpc_pw_sd, inflpc_pw_ad],
    ["MSD", "MAD", "PW-MSD", "PW-MAD"],
    "inflpc_using_translated_sphere_metrics_histograms",
    viewbins=50,
    xlim=[(-0.05, 0.05), (-0.05, 0.05), (-0.005, 0.005), (-0.05, 0.05)]
)

# 4. TranSph analysis (Translated Sphere)
print("Protein volume")
morphpc_volume_translated = morphpc_translated_summary["protein_int_volume_holes_removal"]
inflpc_volume_translated = inflpc_translated_summary["protein_int_volume_holes_removal"]
volume_decrease_translated = (inflpc_volume_translated - morphpc_volume_translated) / inflpc_volume_translated * 100
volume_decrease_stats_translated = {
    "mean": volume_decrease_translated.mean(),
    "min": volume_decrease_translated.min(),
    "max": volume_decrease_translated.max(),
    "std": volume_decrease_translated.std(),
}

report.append("TranSph Analysis (Translated Sphere):")
report.append(f"Volume Decrease Stats: {volume_decrease_stats_translated}")
report.append("")


# Plot histogram for volume decrease
fig, ax = plt.subplots()
ax.hist(volume_decrease_translated, bins=50, edgecolor="black")
ax.set_xlabel("Volume Reduction (%)")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "morphpc_volume_decrease_translated_histogram.pdf")


# # Depth indices correlation (Translated Sphere)
# print("InflPC vs MorphPC Depth Indices Correlation")
# correlation_translated = np.corrcoef(morphpc_translated_di, inflpc_translated_di)[0, 1]
# report.append("Depth Indices Correlation (Translated Sphere):")
# report.append(f"Correlation: {correlation_translated}")
# report.append("")

# # Scatter plot (Translated Sphere)
# fig, ax = plt.subplots()
# ax.scatter(morphpc_translated_di, inflpc_translated_di, s=1)
# plt.rcParams['text.usetex'] = True
# ax.set_xlabel(r"\textsc{MorphPC} Depth Indices")
# ax.set_ylabel(r"\textsc{InflPC} Depth Indices")
# ax.xaxis.label.set_size(16)
# ax.yaxis.label.set_size(16)
# ax.set_xlim(0, 2)
# ax.set_ylim(0, 2)
# ax.tick_params(axis='both', labelsize=13)
# fig.tight_layout()
# fig.savefig(os.path.join(OUTPUT_FOLDER, "depth_indices_translated_scatter.png"), dpi=300)
# plt.close(fig)

# Histogram (Translated Sphere)
fig, ax = plt.subplots()
ax.hist(morphpc_translated_di, bins=50, alpha=0.5, label="MorphPC")
ax.hist(inflpc_translated_di, bins=50, alpha=0.5, label="InflPC")
ax.set_xlabel("Depth Indices")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_xlim(0, 2)
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=11) #, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2)  # Place legend above plot, outside
fig.tight_layout()
save_plot(fig, "depth_indices_translated_histogram.pdf")

# Absolute difference in reference radius (Translated Sphere)
reference_radius_diff_translated =  inflpc_translated_summary["reference_radius"] - morphpc_translated_summary["reference_radius"]

# Histogram of the reference radius difference (Translated Sphere)
fig, ax = plt.subplots()
ax.hist(reference_radius_diff_translated, bins=70, edgecolor="black")
ax.set_xlabel("Reference Radius Difference")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_xlim(-2.5, 5.0)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "reference_radius_diff_translated_histogram.pdf")

# 5. N vs n correlation
print("N vs n correlation")
# Unaligned case
correlation_n_n_unaligned = np.corrcoef(unaligned_summary["N"], unaligned_summary["n"])[0, 1]

# Scatter plot without fitting line (Unaligned)
fig, ax = plt.subplots()
ax.scatter(unaligned_summary["N"], unaligned_summary["n"], color="blue")
ax.set_title("Unaligned: N vs n")
ax.set_xlabel("N")
ax.set_ylabel("n")
save_plot(fig, "unaligned_n_vs_n_scatter.pdf")

# Scatter plot with fitting line (Unaligned)
fig, ax = plt.subplots()
ax.scatter(unaligned_summary["N"], unaligned_summary["n"], color="blue")
m, b = np.polyfit(unaligned_summary["N"], unaligned_summary["n"], 1)
ax.plot(unaligned_summary["N"], m * unaligned_summary["N"] + b, color="red", label=f"Fit: y={m:.2f}x+{b:.2f}")
ax.set_xlabel("N (Number of Atoms)")
ax.set_ylabel("n (Number of Voxels)")
ax.set_xticks([0, 1e4, 2e4, 3e4, 4e4])
ax.legend(fontsize=11)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "unaligned_n_vs_n_scatter_with_fit.pdf")

# Aligned case
correlation_n_n_aligned = np.corrcoef(morphpc_translated_summary["N"], morphpc_translated_summary["n"])[0, 1]

# Scatter plot without fitting line (Aligned)
fig, ax = plt.subplots()
ax.scatter(morphpc_translated_summary["N"], morphpc_translated_summary["n"], color="green")
ax.set_title("Aligned: N vs n")
ax.set_xlabel("N")
ax.set_ylabel("n")
save_plot(fig, "aligned_n_vs_n_scatter.pdf")

# Scatter plot with fitting line (Aligned)
fig, ax = plt.subplots()
ax.scatter(morphpc_translated_summary["N"], morphpc_translated_summary["n"], s=2, label="(N, n) pairs")
m, b = np.polyfit(morphpc_translated_summary["N"], morphpc_translated_summary["n"], 1)
ax.plot(morphpc_translated_summary["N"], m * morphpc_translated_summary["N"] + b, label="Fitting line", color="red")
ax.set_xlabel("N (Number of Atoms)")
ax.set_ylabel("n (Number of Voxels)")
ax.set_ylim(0, 0.5e8)
ax.set_xticks([0, 1e4, 2e4, 3e4, 4e4])
ax.legend(fontsize=11)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "aligned_n_vs_n_scatter_with_fit.pdf")

print("Execution time")
# Create a new folder for execution time plots
execution_time_folder = os.path.join(OUTPUT_FOLDER, "execution_time")
os.makedirs(execution_time_folder, exist_ok=True)

# Calculate execution time for each case
cases = {
    "MorphPC Translated": morphpc_translated_summary,
    "InflPC Translated": inflpc_translated_summary,
    "MorphPC Reconstructed": morphpc_reconstructed_summary,
    "InflPC Reconstructed": inflpc_reconstructed_summary,
    "Unaligned": unaligned_summary,
}

for case_name, summary in cases.items():
    # Calculate execution time
    summary["execution_time"] = (
        summary["t_alignment"] +
        summary["t_discretization"] +
        summary["t_fill_space"] +
        summary["t_holes_removal"] +
        summary["t_reference_radius"] +
        summary["t_indexes_computation"]
    )
    
    # Scatter plot of N vs execution time
    fig, ax = plt.subplots()
    ax.scatter(summary["N"], summary["execution_time"], label=case_name, s=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Execution Time")
    ax.set_title(f"{case_name}: N vs Execution Time")
    save_plot(fig, os.path.join(execution_time_folder, f"{case_name.lower().replace(' ', '_')}_N_vs_execution_time.pdf"))
    
    # Scatter plot of n vs execution time
    fig, ax = plt.subplots()
    ax.scatter(summary["n"], summary["execution_time"], label=case_name, s=2)
    ax.set_xlabel("n")
    ax.set_ylabel("Execution Time")
    ax.set_title(f"{case_name}: n vs Execution Time")
    save_plot(fig, os.path.join(execution_time_folder, f"{case_name.lower().replace(' ', '_')}_n_vs_execution_time.pdf"))

# Combined scatter plot of execution time vs N for Translated+InflPC and Translated+MorphPC
fig, ax = plt.subplots()
inflpc_reconstructed_summary
ax.scatter(inflpc_reconstructed_summary["N"], inflpc_reconstructed_summary["execution_time"], label="ReconSph+InflPC", s=4, color="tab:blue")
ax.scatter(morphpc_translated_summary["N"], morphpc_translated_summary["execution_time"], label="TranSph+MorphPC", s=4, color="tab:red", marker="^")
ax.set_xlabel("N (Number of Atoms)")
ax.set_ylabel("Execution Time (seconds)")
ax.set_ylim(0, 180)
ax.set_xticks([0, 1e4, 2e4, 3e4, 4e4])
ax.legend(fontsize=11)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "execution_time_vs_n_atoms_combined.pdf")

# Combined scatter plot of execution time vs n for Translated+InflPC and Translated+MorphPC
fig, ax = plt.subplots()
ax.scatter(inflpc_reconstructed_summary["n"], inflpc_reconstructed_summary["execution_time"], label="ReconSph+InflPC", s=4, color="tab:blue")
ax.scatter(morphpc_translated_summary["n"], morphpc_translated_summary["execution_time"], label="TranSph+MorphPC", s=4, color="tab:red", marker="^")
ax.set_xlabel("n (Number of Voxels)")
ax.set_ylabel("Execution Time (seconds)")
ax.set_ylim(0, 180)
ax.set_xlim(0, 0.5e8)
ax.legend(fontsize=11)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
save_plot(fig, "execution_time_vs_n_voxels_combined.pdf")

# Calculate total execution time and execution time for index computation
morphpc_transph_total_time = morphpc_translated_summary["execution_time"].sum() / 3600  # Convert to hours
morphpc_transph_index_time = morphpc_translated_summary["t_indexes_computation"].sum() / 3600  # Convert to hours

inflpc_reconsph_total_time = inflpc_reconstructed_summary["execution_time"].sum() / 3600  # Convert to hours
inflpc_reconsph_index_time = inflpc_reconstructed_summary["t_indexes_computation"].sum() / 3600  # Convert to hours

# Calculate speedup
total_time_speedup = inflpc_reconsph_total_time / morphpc_transph_total_time
index_time_speedup = inflpc_reconsph_index_time / morphpc_transph_index_time

# Append results to the report
report.append("Execution Time Analysis:")
report.append(f"Total Execution Time (MorphPC+TranSph): {morphpc_transph_total_time:.2f} hours")
report.append(f"Total Execution Time (InflPC+ReconSph): {inflpc_reconsph_total_time:.2f} hours")
report.append(f"Speedup (Total Execution Time): {total_time_speedup:.2f}")
report.append("")
report.append(f"Index Computation Time (MorphPC+TranSph): {morphpc_transph_index_time:.2f} hours")
report.append(f"Index Computation Time (InflPC+ReconSph): {inflpc_reconsph_index_time:.2f} hours")
report.append(f"Speedup (Index Computation Time): {index_time_speedup:.2f}")
report.append("")

# 8. ResidueDepth correlation
print("ResidueDepth")

residuedepth_ids = [
    "1d7t", "2a3c", "1hp2", "7ljs", "6c6t", "6svm",
    "3o6x", "6yvd", "3h7l", "4hnt", "4y2l", "4iuu",
    "5thk", "6xo4", "4ysy", "5byv", "7e8b", "1mnq",
    "6p6p", "3dwl", "2j3m", "7rth", "5xon", "2oby",
    "2aaz", "5jz7", "6vk6", "6zfp", "4f49", "5mxd"
]

# morphpc_translated_depth = inflpc_translated_depth

# Filter MorphPC translated depth data for ResidueDepth IDs
filtered_morphpc_translated_depth = morphpc_translated_depth[morphpc_translated_depth["pdb_code"].isin(residuedepth_ids)]
filtered_residuedepth_data = residuedepth_data[residuedepth_data["pdb_code"].isin(residuedepth_ids)]
# filtered_residuedepth_data = residuedepth_data
# filtered_morphpc_translated_depth = inflpc_reconstructed_depth

# Normalize residue depth data
# filtered_residuedepth_data["depth_normalized"] = filtered_residuedepth_data.groupby("pdb_code")["depth"].transform(
#     # lambda x: 2 * (1 - (x - x.min()) / (x.max() - x.min())))
#     lambda x: 2 * (x) / (x.max()))

print("Normalize ResidueDepth data")
for pdb_ids in filtered_residuedepth_data["pdb_code"].unique():
    pdb_data = filtered_residuedepth_data[filtered_residuedepth_data["pdb_code"] == pdb_ids]
    if len(pdb_data) > 0:
        min_depth = pdb_data["depth"].min()
        max_depth = pdb_data["depth"].max()
        print("Max depth for PDB ID", pdb_ids, ":", max_depth)
        print("Min depth for PDB ID", pdb_ids, ":", min_depth)
        normalized_depth = 2 * (pdb_data["depth"]) / (max_depth)
        filtered_residuedepth_data.loc[filtered_residuedepth_data["pdb_code"] == pdb_ids, "depth_normalized"] = normalized_depth

all_data = pd.merge(
    filtered_residuedepth_data,
    filtered_morphpc_translated_depth,
    on=["pdb_code", "index"],
    suffixes=("_residuedepth", "_morphpc_translated"),
    how="inner"
)

sorted_data = all_data.sort_values(by="depth_normalized").reset_index(drop=True)
sorted_data_first_10_percent = sorted_data.head(int(len(sorted_data) * 0.1))

# Plot scatter plot
fig, ax = plt.subplots()
ax.scatter(all_data["depth_normalized"],
           all_data["depth_morphpc_translated"], s=1, alpha=0.01)

# Use hexbin plot for better visualization with many points
# hb = ax.hexbin(
#     all_data["depth_normalized"],
#     all_data["depth_morphpc_translated"],
#     gridsize=50,
#     cmap="viridis",
#     mincnt=1000)

# cb = fig.colorbar(hb, ax=ax)
# cb.set_label("Counts")

ax.set_xlabel("Normalized DEPTH")
ax.set_ylabel("SADIC depth index")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "edtsurf_scatter.png"), dpi=300)
plt.close(fig)

print("Scatter plot saved as edtsurf_scatter.png")

fig, ax = plt.subplots()
bins = np.linspace(0, 2, 100)
ax.hist(sorted_data_first_10_percent["depth_normalized"], bins=bins, alpha=0.5, label="MorphPC")
ax.hist(sorted_data_first_10_percent["depth_morphpc_translated"], bins=bins, alpha=0.5, label="InflPC")
ax.set_xlabel("Depth Indices")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_xlim(0, 2)
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
save_plot(fig, "residuedepth_histogram_10.pdf")

fig, ax = plt.subplots()
ax.hist(all_data["depth_normalized"], bins=bins, alpha=0.5, label="DEPTH (Normalized)")
ax.hist(all_data["depth_morphpc_translated"], bins=bins, alpha=0.5, label="SADIC depth index")
ax.set_xlabel("Depth Indices")
ax.set_ylabel("Frequency")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.set_xlim(0, 2)
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
save_plot(fig, "residuedepth_histogram_all.pdf")

# Compute correlation between normalized depth data and depth indices (MorphPC, Translated, Aligned)
correlation_normalized_morphpc = np.corrcoef(all_data["depth_normalized"], all_data["depth_morphpc_translated"])[0, 1]
spearman_correlation_normalized_morphpc, _ = spearmanr(all_data["depth_normalized"], all_data["depth_morphpc_translated"])
report.append(f"Correlation (Normalized Depth vs MorphPC Translated Depth): {correlation_normalized_morphpc}")
report.append(f"Spearman Correlation (Normalized Depth vs MorphPC Translated Depth): {spearman_correlation_normalized_morphpc}")

# Compute protein-wise correlations
protein_wise_correlations = []
for pdb_code, group in tqdm(all_data.groupby("pdb_code")):
    morphpc_depth_values = all_data.loc[
        all_data["pdb_code"] == pdb_code, "depth_morphpc_translated"
    ].values
    correlation = np.corrcoef(group["depth_normalized"].values, morphpc_depth_values)[0, 1]
    protein_wise_correlations.append(correlation)
protein_wise_correlations = pd.Series(protein_wise_correlations)

# Plot histogram of protein-wise correlations
fig, ax = plt.subplots()
ax.hist(protein_wise_correlations, bins=30, color="purple", edgecolor="black")
ax.set_title("Protein-wise Correlations (Normalized Depth vs MorphPC Translated Depth)")
ax.set_xlabel("Correlation Coefficient")
ax.set_ylabel("Frequency")
save_plot(fig, "protein_wise_correlations_histogram.pdf")

# Compute statistics for protein-wise correlations
protein_wise_correlation_stats = {
    "min": protein_wise_correlations.min(),
    "max": protein_wise_correlations.max(),
    "avg": protein_wise_correlations.mean(),
    "std": protein_wise_correlations.std(),
}

report.append("Protein-wise Correlation Stats (Normalized Depth vs MorphPC Translated Depth):")
report.append(f"Min: {protein_wise_correlation_stats['min']}")
report.append(f"Max: {protein_wise_correlation_stats['max']}")
report.append(f"Avg: {protein_wise_correlation_stats['avg']}")
report.append(f"Std: {protein_wise_correlation_stats['std']}")
report.append("")

# 9. Line plot
# Sort depth indices for each algorithm
sorted_residuedepth = np.sort(all_data["depth_normalized"])[::-1]
sorted_morphpc_translated = np.sort(morphpc_translated_depth["depth"])
sorted_inflpc_translated = np.sort(inflpc_translated_depth["depth"])

# Create x-axis values
x_values_residuedepth = np.linspace(0, 2e7, len(sorted_residuedepth))
x_values_morphpc_translated = np.linspace(0, 2e7, len(sorted_morphpc_translated))
x_values_inflpc_translated = np.linspace(0, 2e7, len(sorted_inflpc_translated))

# Plot the sorted depth indices
fig, ax = plt.subplots() #figsize=(8, 6))
ax.plot(x_values_residuedepth, sorted_residuedepth, label="DEPTH (Normalised)", color="blue")
ax.plot(x_values_morphpc_translated, sorted_morphpc_translated, label="MorphPC+TranSph", color="green")
ax.plot(x_values_inflpc_translated, sorted_inflpc_translated, label="InflPC+TranSph", color="red")

# Plot a vertical dashed line at 10% from the right
ax.axvline(x=0.9 * x_values_residuedepth[-1], color="black", linestyle="--", label="10% Threshold")

# ax.set_title("Sorted Depth Indices Comparison")
ax.set_xlabel("Atom Index")
ax.set_ylabel("Depth")
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=13)
ax.legend(fontsize=11) #, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2)  # Place legend above plot, outside
fig.tight_layout()
save_plot(fig, "sorted_depth_indices_comparison.pdf")

save_report(report, "analysis_report.txt")
print("Analysis complete. Results saved to:", OUTPUT_FOLDER)