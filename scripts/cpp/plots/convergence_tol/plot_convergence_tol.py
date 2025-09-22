import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to data folder
data_folder = "data"

# Find all files matching tN_hf_energies.csv
files = sorted(glob.glob(os.path.join(data_folder, "t*_hf_energies.csv")))

if not files:
    raise FileNotFoundError("No files found in 'data/' with pattern tN_hf_energies.csv")

# Read last file to get E_ref
last_file = files[-1]
last_df = pd.read_csv(last_file, header=None)
E_ref = last_df.iloc[-1, 0]

# Determine minimum number of entries across files
min_len = min(len(pd.read_csv(f, header=None)) for f in files)

# Create plot
plt.figure(figsize=(8,6))

# Darker colormap
colors = plt.cm.cividis(np.linspace(0, 1, len(files)))

offset = 2
for i, f in enumerate(files):
    df = pd.read_csv(f, header=None)
    energies = df.iloc[:min_len, 0]  # truncate to min length
    iterations = np.arange(1, min_len + 1)
    y = np.abs(energies / E_ref - 1)+1e-10
    label = "$10^{-" + str(i + offset) + "}$"
    plt.plot(iterations, y, label=label, linewidth=2)

# Labels and legend
plt.xlabel("Iteration number", fontsize=12)
plt.ylabel(r"Relative error [-]", fontsize=12)
plt.yscale("log")
plt.legend(title="CG tolerance")
plt.grid(True, linestyle="--", alpha=0.5, which="both")

plt.tight_layout()
plt.show()
