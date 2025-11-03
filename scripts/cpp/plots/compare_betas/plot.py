import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# --- CONFIGURATION ---
# Automatically load all JSONs in the current folder:
files = sorted(glob.glob("*.json"))  # adjust path if needed

plt.figure(figsize=(8, 6))

for filename in files:
    with open(filename, 'r') as f:
        content = json.load(f)

    data = content["data"]

    # Extract and sort by beta
    beta = np.array([entry["beta"] for entry in data])
    Eint = np.array([entry["Eint"] for entry in data])
    sort_idx = np.argsort(beta)
    beta = beta[sort_idx]
    Eint = Eint[sort_idx]

    print(f"file name : {filename}")
    # Smooth interpolation using cubic spline
    beta_smooth = np.linspace(beta.min(), beta.max(), 200)
    spline = make_interp_spline(beta, Eint, k=1)
    Eint_smooth = spline(beta_smooth)

    # Find minimum of smoothed curve
    idx_min = np.argmin(Eint_smooth)
    beta_min = beta_smooth[idx_min]
    Eint_min = Eint_smooth[idx_min]

    # Plot
    label_name = content["name"]
    plt.plot(beta_smooth, Eint_smooth, label=f"{label_name}:  {label_name}$_{'\\text{min}'}$={beta_min:.3f}")
    plt.scatter(beta_min, Eint_min, marker='D', zorder=5, s=20)

fontsize = 11
plt.xlabel('$\\beta_2$', fontsize=fontsize+3)
plt.ylabel('$E_{\\text{int}}$', fontsize=fontsize+3)
plt.legend(fontsize=fontsize, loc="best", frameon=False)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

