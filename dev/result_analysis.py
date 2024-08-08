import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output2.csv')

print(df.head())
print(df.columns)

# remove the entries where the column "resolution" is the string "ERROR"
initial_rows = len(df)
df = df[df["resolution"] != "ERROR"]
final_rows = len(df)
print(f"Removed {initial_rows - final_rows}/{initial_rows} rows, remaining {final_rows}.")

interesting_columns = ["t1", "t2", "t3", "t4", "N", "n", "p_4_max"]
alias = ["discretization time", "holes removal time", "reference radius time", "indexes computation time", "N", "n", "p_max"]

df = df[interesting_columns]
df.columns = alias

print(df.head())

df["N"].plot(kind="hist", title="N distribution")

plots = [
    ("N", "n"),
    ("N", "p_max"),
    ("n", "p_max"),
    ("N", "discretization time"),
    ("N", "holes removal time"),
    ("N", "reference radius time"),
    ("N", "indexes computation time"),
    ("n", "discretization time"),
    ("n", "holes removal time"),
    ("n", "reference radius time"),
    ("n", "indexes computation time")
]

for plot in plots:
    df = df.sort_values(by=plot[0])
    df.plot(x=plot[0], y=plot[1], kind="scatter", title=f"{plot[0]} vs {plot[1]}")

plt.show()