import numpy as np
import matplotlib.pyplot as plt

def plot_ho_2d():
    eigenvalues = np.loadtxt("output/ho_2d_eigenvalues.txt")
    eigenvectors = np.loadtxt("output/ho_2d_eigenvectors.txt")
    x = np.loadtxt("output/ho_2d_x.txt")


