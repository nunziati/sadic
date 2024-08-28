import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('summary.csv')
folder = os.path.join("experiments", )

print(df.head())
print(df.columns)

# remove the entries where the column "resolution" is the string "ERROR"
initial_rows = len(df)
df = df[df["resolution"] != "ERROR"]
final_rows = len(df)
print(f"Removed {initial_rows - final_rows}/{initial_rows} rows, remaining {final_rows}.")

print(df.head())

df["N"].plot(kind="hist", title="N distribution")

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
    # "depth_indexes_path",
    "protein_int_volume_discretization",
    "protein_int_volume_space_fill",
    "connected_components",
    "protein_int_volume_holes_removal",
    # "p_disc_path",
    "p_disc_min",
    "p_disc_max",
    "p_disc_avg",
    "p_disc_med",
    "p_disc_std",
    # "disc_voxel_operations_map_path",
    # "p_idx_path",
    "p_idx_min",
    "p_idx_max",
    "p_idx_avg",
    "p_idx_med",
    "p_idx_std",
    #"idx_voxel_operations_map_path",
    )

df["N*p_max/n"] = df["N"] * df["p_idx_max"] / df["n"]
df["N*p_avg/n"] = df["N"] * df["p_idx_avg"] / df["n"]

plots = [
    ("N", "n"),
    ("N", "t_alignment"),
    ("N", "t_discretization"),
    ("N", "t_fill_space"),
    ("N", "t_holes_removal"),
    ("N", "t_reference_radius"),
    ("N", "t_indexes_computation"),
    ("n", "t_alignment"),
    ("n", "t_discretization"),
    ("n", "t_fill_space"),
    ("n", "t_holes_removal"),
    ("n", "t_reference_radius"),
    ("n", "t_indexes_computation"),
    ("N", "reference_radius"),
    ("n", "reference_radius"),
    ("N", "p_disc_max"),
    ("n", "p_disc_max"),
    ("N", "p_idx_max"),
    ("n", "p_idx_max"),
]

"""for plot in plots:
    df = df.sort_values(by=plot[0])
    if plot[0] == "n":
        df.plot(x=plot[0], y=plot[1], kind="scatter", title=f"{plot[0]} vs {plot[1]}", xlim=(0, 1e8))
    else:
        df.plot(x=plot[0], y=plot[1], kind="scatter", title=f"{plot[0]} vs {plot[1]}")

df = df.sort_values(by="N")
df.plot(x="N", ylim=(-1, 20), y=["t_alignment", "t_discretization", "t_fill_space", "t_holes_removal", "t_reference_radius", "t_indexes_computation"], kind="line", title="Time vs N")

df = df.sort_values(by="n")
df.plot(x="n", ylim=(-1, 20), y=["t_alignment", "t_discretization", "t_fill_space", "t_holes_removal", "t_reference_radius", "t_indexes_computation"], kind="line", title="Time vs n")
"""

df["N*p_max/n"].plot(kind="hist", title="N*p_max/n distribution")
df["N*p_avg/n"].plot(kind="hist", title="N*p_avg/n distribution")

plt.show()