import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair
import util as u

a1 = 5
a2 = 5

def matsetup_pot_well(n1, n2):
    A = np.zeros(n1*n2, n1*n2)
    xs = np.linspace(-a2, a2, n2)
    ts = np.linspace(0, a1, n1)
    h1 = 2*a1/n1
    h2 = 2*a2/n2
    np.fill_diagonal(A[:n1, :n1], 0)
    np.fill_diagonal(A[1:n1, :n1], -0.5/h1)
    np.fill_diagonal(A[:n1, 1:n1])
    for i in range(n1):

    return A, xs, ts

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

