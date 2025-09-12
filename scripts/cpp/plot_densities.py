import numpy as np
import matplotlib.pyplot as plt
import math
import plotly
import plotly.graph_objects as go

name = "be"
density = np.genfromtxt("output/def/density.csv")
#density = np.genfromtxt("output/density.csv")
n = 40
n2 = n // 2
a = 12

mat_or = density.reshape((n, n, n))

#mat = mat[:, :, n // 2]
mat = mat_or[:, :, n2]
#mat = mat[n // 2, :, :]

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)
z = np.linspace(-a, a, n)

#rho_z = mat_or.sum(axis=0)
#z = np.linspace(-a, a, n)
#plt.plot(z, rho_z)
#plt.title("Densità integrata su x e y (funzione di z)")
#plt.xlabel("z [fm]")
#plt.ylabel(r"$\int dx\,dy\, \rho(x,y,z)$")
#plt.grid()
#plt.show()

#rho_z = mat[:, 10, 10]

#mat = mat[:, 0, :]
#mat = mat[n // 2, :, :]

max = max(mat.flatten())
threshold = max / 2

X, Y = np.meshgrid(x, y)

contour = plt.contourf(X, Y, mat, cmap='viridis', levels = 10)
plt.colorbar(contour)
plt.title("Total particle density")
plt.xlabel('x [fm]')
plt.ylabel("y [fm]")
major_ticks = np.arange(-a, a, a/5)
plt.grid(linewidth = 0.2)
plt.xticks(major_ticks)
plt.yticks(major_ticks)

plt.show()
