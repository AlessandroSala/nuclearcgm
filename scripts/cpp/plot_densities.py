import numpy as np
import matplotlib.pyplot as plt

density = np.genfromtxt("output/density.csv")
n = 40
a = 13

def den(i, j):
    idx = i*n + j + n//2
    return  density[idx]

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)

mat = density.reshape((n, n, n))
mat = mat[:, :, 0]

X, Y = np.meshgrid(x, y)

contour = plt.contourf(X, Y, mat, cmap='viridis', levels = 100)
plt.colorbar(contour)
plt.title("Neutron particle density $^{16}$O")
plt.xlabel("x [fm]")
plt.ylabel("y [fm]")
major_ticks = np.arange(-a, a, a/5)
plt.grid(linewidth = 0.2)
plt.xticks(major_ticks)
plt.yticks(major_ticks)

plt.show()


