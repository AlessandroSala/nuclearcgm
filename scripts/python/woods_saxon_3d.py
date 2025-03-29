import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair_unalloc, find_eigenpair
from nuc_constants import *
import util as u
folder_name = "woods_saxon"
A = 16
n = 19

R = 1.27*A**(1/3)
diff = 0.67
V0 = 57 # Assuming Z = N


a = 12
xs = np.linspace(-a, a, n)
ys = np.linspace(-a, a, n)
zs = np.linspace(-a, a, n)
h = 2*a/n

def pot(x, y, z):
    return -V0*(1+np.exp((np.sqrt(x**2+y**2+z**2) - R)/diff))**-1
def idx(i, j, k):
    return i+n*(j+n*(k))

def A(i, j, k, i1, j1, k1):
    if i1 == i and j1 == j and k==k1:
        return (pot(xs[i], ys[j], zs[k])*C*h**2-6)/(C*h**2)
    elif (i==i1 and k==k1 and (j==j1+1 or j==j1-1)) or (j==j1 and k==k1 and (i==i1+1 or i==i1-1)) or (i==i1 and j==j1 and (k==k1+1 or k==k1-1)) :
        return 1/(C*h**2)

    return 0
def reverse_index(index):
    k = n // (n * n)                # Calcola k
    j = (n % (n * n)) // n        # Calcola j
    i = n % n                         # Calcola i
    return i, j, k
def A_callable(a, b):
    return A(*reverse_index(a),*reverse_index(b) )

def matsetup_oscillator(n):
    mat = np.zeros((n**3, n**3))
    #pot = 0
    for i in range(n):
        for j in range(n):
            for i1 in range(n):
                for j1 in range(n):
                    for k in range(n):
                        for k1 in range(n):
                            n0 = idx(i, j, k)
                            n1 = idx(i1, j1, k1)
                            mat[n0, n1] = A(i, j, k, i1, j1, k1)

    return mat

mat = matsetup_oscillator(n)
print("Matrix generated")
res = find_eigenpair(mat, np.random.rand(n**3), tol = 1e-30, n_max = 5000, verbose=True)
np.savetxt(f"output/{folder_name}/eigenvectors.txt", res[0])
np.savetxt(f"output/{folder_name}/eigenvalues.txt", np.array([res[1]]))
np.savetxt(f"output/{folder_name}/x.txt", xs)
np.savetxt(f"output/{folder_name}/y.txt", ys)
np.savetxt(f"output/{folder_name}/y.txt", zs)
#gauss =np.linalg.eig(mat[0]) 
#min_eig = np.argmin(np.abs(gauss[0]))
#
#print(f"Error: {res[1] - gauss[0][min_eig]}")
print(f"GS Energy value: {res[1]} MeV") 
E_real = 31
print(f"Error GS: {round((res[1]/E_real - 1)*100, 2)}%")
y = u.positive_vector(u.normalize(res[0]))

plt.plot(xs, u.positive_vector(u.normalize (res[0][5*n:6*n])))
plt.show()
#res_2 = find_eigenpair_constrained(mat[0], u.normalize(guess), u.normalize(res[0]), tol = 1e-22, c= 1000)
#plt.plot(x, y**2, label="$|\\psi|^2$")
#plt.xlabel("$x$ [fm]")
#plt.legend()
##plt.plot(x, (u.normalize(res_2[0])))
#plt.grid()
#plt.show()

