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
cutoff = 7
print("h:", h)

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)
z = np.linspace(-a, a, n)
r = np.sqrt(x**2 + y**2 + z**2)
# Create 3D grids
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt(X**2 + Y**2 + Z**2)
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

        # Load clustering
        density = np.genfromtxt(file_path)
        mat_or = density.reshape((n, n, n))
        mat_or[ r > cutoff] = 0.0
        mat = mat_or[:, :, n // 2]

        #load rho
        rho = np.genfromtxt(os.path.join(root, "density.csv"))
        mat_rho = rho.reshape((n, n, n))
        mat_rho[mat_or < 0.90] = 0.0
        integral = mat_rho.sum()*dV
        print(f"Integral: {integral}")


        # Axes
        x = np.linspace(-a, a, n)
        y = np.linspace(-a, a, n)
        X, Y = np.meshgrid(x, y)

        # Plot
        plt.figure(figsize=(6, 5), constrained_layout=True)
        contour = plt.contourf(X, Y, mat, cmap=cmap, levels=100, vmin = 0.5)
        plt.xlabel('x [fm]')
        plt.ylabel('y [fm]')
        plt.colorbar(label=r"NLF [-]")
        limit = 7
        plt.axis([-limit, limit, -limit, limit])
        major_ticks = np.arange(-limit, limit + 0.1, limit / 5)
        plt.xticks(major_ticks)
        plt.yticks(major_ticks)

        # Save figure
        out_path = os.path.join( "figures/clustering/", f"{folder_name}_localization.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        #np.savetxt(out_path.replace(".pdf", ".txt"), integral)
        plt.close()

        print(f"Saved plot as {out_path}")

print("All done.")

