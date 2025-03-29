import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair
import util as u


h_bar = 6.62607015e-34
m = 9.10938356e-31
a = 5.291772109e-11
ev = 1.60218e-19

def matsetup_pot_well(n):
    A = np.zeros((n, n))
    xs = np.linspace(-a, a, n)
    h = 2*a/n
    C = (-2*m*h**2/h_bar**2)**-1
    np.fill_diagonal(A, -2*C)
    np.fill_diagonal(A[1:, :], C)
    np.fill_diagonal(A[:, 1:], C)
    A[0, -1] = 0
    A[-1, 0] = 0
    return A, xs

n = 1000
mat = matsetup_pot_well(n)
res = find_eigenpair(mat[0], np.random.rand(n), tol = 1e-15)
#gauss =np.linalg.eig(mat[0]) 
#min_eig = np.argmin(np.abs(gauss[0]))
#
#print(f"Error: {res[1] - gauss[0][min_eig]}")
print(f"Energy value: {res[1]/ev} eV") 
E_real = np.pi**2*h_bar**2/(8*a**2*m)
print(f"Error: {round((res[1]/E_real - 1)*100, 2)}%")

x = mat[1]
y = u.positive_vector(u.normalize(res[0]))
y_real = u.positive_vector(u.normalize( np.cos(0.5*np.pi*x/a)))
plt.plot(x, y, label="Approximated solution")
plt.plot(x, y_real, label="Exact solution")
plt.legend()
plt.grid()
#plt.plot(mat[1], u.positive_vector(gauss[1][:, min_eig]))
plt.show()

