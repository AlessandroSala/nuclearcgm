import pandas as pd
import matplotlib.pyplot as plt
import math

folder = "MG24"
# List of file pairs
file_pairs = [
    (f"{folder}/inn1_hf_energies.csv", f"{folder}/inn1_tot_energies.csv"),
    (f"{folder}/inn2_hf_energies.csv", f"{folder}/inn2_tot_energies.csv"),
    (f"{folder}/inn3_hf_energies.csv", f"{folder}/inn3_tot_energies.csv"),
    (f"{folder}/inn4_hf_energies.csv", f"{folder}/inn4_tot_energies.csv"),
    (f"{folder}/inn5_hf_energies.csv", f"{folder}/inn5_tot_energies.csv"),
    (f"{folder}/inn6_hf_energies.csv", f"{folder}/inn6_tot_energies.csv"),
]

# --- Determine the smallest number of data points among all files ---
min_len = min(len(pd.read_csv(f, header=None)) for pair in file_pairs for f in pair)

# Skip the first 4 points
min_len = max(0, min_len - 0)

#max_points = min(75, min_len)  # cap to at most 75

max_points = 80
# --- Reference values from the last pair ---
ref_hf_file, ref_tot_file = file_pairs[-1]
ref_hf = pd.read_csv(ref_hf_file, header=None).iloc[-1, 0]
ref_tot = pd.read_csv(ref_tot_file, header=None).iloc[-1, 0]

# --- Subplot layout: 3 per row ---
n_pairs = len(file_pairs)
ncols = 3
nrows = math.ceil(n_pairs / ncols)

fig, axes = plt.subplots(
    nrows, ncols, figsize=(5*ncols, 4*nrows),
    sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]}
)

# Flatten axes for easy indexing
axes = axes.flatten()

exclude = 5
for i, (f1, f2) in enumerate(file_pairs):
    ax = axes[i]

    # Read the files and skip first 4 points
    y1 = pd.read_csv(f1, header=None)[0][4:4+min_len]  # HF
    y2 = pd.read_csv(f2, header=None)[0][4:4+min_len]  # TOT

    # Take only the last max_points values
    y1 = y1[-max_points:]
    y2 = y2[-max_points:]

    # Normalize separately using ref_hf and ref_tot
    #y1_scaled = abs(y1 / ref_hf - 1)
    #y2_scaled = abs(y2 / ref_tot - 1)
    y1_scaled = abs(y1 / ref_hf - 1)
    y2_scaled = abs(y2 / ref_tot - 1)

    # x starts at 0
    data_len = len(y1)
    x = range(data_len)

    # Plot both curves
    ax.plot(x[:(data_len - exclude)], y1_scaled[:(data_len - exclude)], label=f"Hartree-Fock energy")
    ax.plot(x[:(data_len - exclude)], y2_scaled[:(data_len - exclude)], label=f"Integral energy")

    ax.set_yscale("log")
    #ax.set_title(f"Pair {i+1}")
    ax.grid(True, which="major", ls="--", alpha=0.7)
    ax.legend(fontsize="small", title = f"{i+1} Inverse power step" + ("" if i == 0 else "s"))

# Remove unused subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.text(0.5, 0.09, "Iteration Number", ha="center")
fig.text(0.09, 0.5, "Relative error [-]", va="center", rotation="vertical")
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
plt.show()
