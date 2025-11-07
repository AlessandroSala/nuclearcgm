import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# --- CONFIGURATION ---
# Automatically load all JSONs in the current folder:
files = sorted(glob.glob("*.json"))  # adjust path if needed

plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(6, 6))

Emins = []
Emax = []

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

    Emins.append(Eint_min)
    Emax.append(Eint_smooth[np.argmax(Eint_smooth)])

    # Plot
    label_name = os.path.splitext(os.path.basename(filename))[0]
    plt.plot(beta_smooth, Eint_smooth, label=f"{label_name}: β={beta_min:.3f}, E={Eint_min:.3f}")
    plt.scatter(beta_min, Eint_min, marker='D', zorder=5, s=30)

fontsize = 13
plt.xlabel("$\\beta_2$", fontsize=fontsize)
plt.ylabel("$E$", fontsize=fontsize)
plt.legend(fontsize=10, loc="best", frameon=False)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.ylim(min(Emins)-1, max(Emax)+5)
plt.show()

