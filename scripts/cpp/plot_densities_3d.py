import numpy as np
import plotly.graph_objects as go

density = np.genfromtxt("output/o16/tensor_test_output/fields/density.csv")

n = 30
a = 10

rho = density.reshape((n, n, n))

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)
z = np.linspace(-a, a, n)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

verts, faces, _, _ = marching_cubes(rho, level=0.8 * rho.max())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mesh = Poly3DCollection(verts[faces], alpha=0.7)
mesh = Poly3DCollection(
    verts[faces],
    alpha=0.85,
    linewidths=0.05
)
mesh.set_edgecolor("k")

ax.add_collection3d(mesh)

ax.set_xlim(0, n//2)
ax.set_ylim(0, n//2)
ax.set_zlim(0, n//2)

plt.show()

