import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair
from nuc_constants import *
import util as u
A = 16
a = 10
omega = 41/h_bar/A**(1/3)
n = 10

xs = np.linspace(-a, a, n)
ys = np.linspace(-a, a, n)
h = 2*a/n

def idx(i, j):
    return i+n*(j)

def A(i, j, i1, j1):
    if i1 == i and j1 == j:
        pot = 0.5*2*xs[i]**2*m*omega**2*C*h**2
        return (pot-2)/(C*h**2)
    elif i==j1+1 or i==j1-1:
        return 1/(C*h**2)

    return 0

def matsetup_oscillator(n):
    mat = np.zeros((n**2, n**2))
    #pot = 0
    for i in range(n):
        for j in range(n):
            for i1 in range(n):
                for j1 in range(n):
                    n0 = idx(i, j)
                    n1 = idx(i1, j1)
                    mat[n0, n1] = A(i, j, i1, j1)

    return mat

mat = matsetup_oscillator(n)
print("Matrix generated")
print(f"Is matrix symmetric? {(mat == mat.T).all()}")
res = find_eigenpair(mat, np.random.rand(n**2), tol = 1e-27, n_max = 1000)
#gauss =np.linalg.eig(mat[0]) 
#min_eig = np.argmin(np.abs(gauss[0]))
#
#print(f"Error: {res[1] - gauss[0][min_eig]}")
print(f"GS Energy value: {res[1]} MeV") 
E_real = h_bar * omega * 1
print(f"Error GS: {round((res[1]/E_real - 1)*100, 2)}%")
y = u.positive_vector(u.normalize(res[0]))
x = mat[1]
#plt.plot(x, u.positive_vector(u.normalize (gauss[1][:, min_eig])))
#res_2 = find_eigenpair_constrained(mat[0], u.normalize(guess), u.normalize(res[0]), tol = 1e-22, c= 1000)
#plt.plot(x, y**2, label="$|\\psi|^2$")
#plt.xlabel("$x$ [fm]")
#plt.legend()
##plt.plot(x, (u.normalize(res_2[0])))
#plt.grid()
#plt.show()

