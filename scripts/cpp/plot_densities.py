import numpy as np
import matplotlib.pyplot as plt
import math
import plotly
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

name = "mg"
density = np.genfromtxt("output/o16/tensor_test_output/fields/density.csv")
#density = np.genfromtxt("output/density.csv")
n = 30
n2 = n // 2
a = 10

mat_or = density.reshape((n, n, n))

#mat = mat[:, :, n // 2]
mat = mat_or[:, :, n2]
#mat = mat[n // 2, :, :]

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)
z = np.linspace(-a, a, n)

#rho_z = mat_or.sum(axis=2)
#z = np.linspace(-a, a, n)
#plt.plot(z, rho_z)
#plt.title("Densità integrata su x e y (funzione di z)")
#plt.xlabel("z [fm]")
#plt.ylabel(r"$\int dx\,dy\, \rho(x,y,z)$")
#plt.grid()
#plt.show()
#
X, Y = np.meshgrid(x, y)

custom_cmap = cm.get_cmap("jet").copy()
custom_cmap.set_under("white")  # color for values below vmin

from scipy.ndimage import gaussian_filter

jet = plt.cm.get_cmap('jet')

# Colors to blend in at the low end (bottom of the spectrum)
# We start with white and transition to light blue before hitting the main jet range.
custom_low_colors = ["white"] # White, Light Cyan, Light Blue

# --- 2. Blend the custom low colors with the upper part of 'jet' ---
# We'll skip the darkest blues of the original jet map (e.g., skip the first 10%).
# np.linspace(0.1, 1, 200) takes colors from 10% to 100% of the original jet.
n_jet_colors = 200
upper_jet_colors = list(jet(np.linspace(0.1, 1, n_jet_colors)))

# Combine the custom low colors with the rest of the jet map
all_colors = custom_low_colors + upper_jet_colors

# --- 3. Create the new custom colormap ---
colors = ["blue", "red"]

# Create the custom colormap
# Matplotlib will smoothly interpolate the color in the middle
blue_to_red_cmap = LinearSegmentedColormap.from_list(
    "BlueRedGradient", # A name for your custom map
    colors
)
custom_jet_fade = LinearSegmentedColormap.from_list("custom_jet_fade", all_colors)

#cmap = cm.get_cmap('jet')
cmap = custom_jet_fade
#cmap = blue_to_red_cmap
cmap.set_under('white')
rho_min = np.max(mat.flatten()) / 10

contour = plt.contourf(X, Y, mat, cmap=cmap, levels = 100, vmin = rho_min, vmax = 0.18)
#plt.colorbar(contour)
plt.xlabel('x [fm]')
plt.ylabel("y [fm]")
plt.colorbar(label=f"Particle density [fm$^{{-3}}$]")
limit = 7
plt.axis([-limit, limit, -limit, limit])
major_ticks = np.arange(-limit, limit+0.1, limit/5)
#plt.grid(linewidth = 0.2)
plt.xticks(major_ticks)
plt.yticks(major_ticks)

plt.show()

rho = mat_or
rho_max = rho.max()

# livello di isosuperficie: ad esempio 90% del massimo
iso_level = 0.9 * rho_max

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

