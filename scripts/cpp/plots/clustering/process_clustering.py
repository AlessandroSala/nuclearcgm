import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# --- Constants ---
a = 10
n = 60
h = 20 / (n - 1)
dV = h**3
print("h:", h)

# --- Custom colormap ---
jet = plt.cm.get_cmap('jet')
custom_low_colors = ["white"]
n_jet_colors = 200
upper_jet_colors = list(jet(np.linspace(0.1, 1, n_jet_colors)))
all_colors = custom_low_colors + upper_jet_colors
custom_jet_fade = LinearSegmentedColormap.from_list("custom_jet_fade", all_colors)
cmap = custom_jet_fade
cmap.set_under('white')

# --- Paths ---
base_dir = "output/ne20_clustering"

# --- Loop through all subfolders ---
for root, dirs, files in os.walk(base_dir):
    if "C.csv" in files:
        folder_name = os.path.basename(root)
        file_path = os.path.join(root, "C.csv")

        print(f"Processing {folder_name}...")

        # Load density
        density = np.genfromtxt(file_path)
        mat_or = density.reshape((n, n, n))
        mat = mat_or[:, :, n // 2]

        # Axes
        x = np.linspace(-a, a, n)
        y = np.linspace(-a, a, n)
        X, Y = np.meshgrid(x, y)

        # Plot
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, mat, cmap=cmap, levels=100)
        plt.xlabel('x [fm]')
        plt.ylabel('y [fm]')
        plt.colorbar(label=r"Particle density [fm$^{-3}$]")
        limit = 7
        plt.axis([-limit, limit, -limit, limit])
        major_ticks = np.arange(-limit, limit + 0.1, limit / 5)
        plt.xticks(major_ticks)
        plt.yticks(major_ticks)
        plt.title(folder_name)

        # Save figure
        out_path = os.path.join( "figures/clustering/", f"{folder_name}_localization.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

        print(f"Saved plot as {out_path}")

print("All done.")

