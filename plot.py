import numpy as np
import matplotlib.pyplot as plt
import util as u

def plot_ho_2d():
    eigenvalues = np.loadtxt("output/ho_2d/eigenvalues.txt")
    eigenvectors = u.normalize(np.loadtxt("output/ho_2d/eigenvectors.txt"))
    
    xs = np.loadtxt("output/ho_2d/x.txt")
    ys = np.loadtxt("output/ho_2d/y.txt")
    n_x, n_y = xs.shape[0], ys.shape[0]  # Numero di punti lungo x e y n_x
    
    # Creazione della figura 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xs, ys)

    Z = eigenvectors.reshape((n_x, n_y)) 
    # Creazione della superficie
    ax.plot_surface(X, Y, Z**2, cmap='viridis')

    # Label sugli assi
    ax.set_xlabel('X [fm]')
    ax.set_ylabel('Y [fm]')
    ax.set_zlabel('$|\\psi|^2$')

    plt.show()

def plot_ws():
    eigenvectors = u.normalize(np.loadtxt("output/wd_3d/eigenvectors_1.txt"))
    
    xs = np.loadtxt("output/wd_3d/x.txt")
    
    n = xs.shape[0]

    plt.plot(xs, eigenvectors[:n]**2)
    plt.grid()
    plt.xlabel("$(x, 0, 0)$ [fm]")
    plt.ylabel("$|\\psi|^2$")
    plt.title("Woods-Saxon, Ground state wavefunction")

    plt.show()

if __name__ == "__main__":
    plot_ws()