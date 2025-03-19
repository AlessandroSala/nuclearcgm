import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair
from constants import *
import util as u

a = 5e-2
omega = 10

def matsetup_oscillator(n):
    A = np.zeros((n, n))
    xs = np.linspace(-a, a, n)
    h = 2*a/n
    pot = 0.5*xs**2*m*omega**2*C*h**2
    #pot = 0
    np.fill_diagonal(A, (pot-2)/(C*h**2))
    np.fill_diagonal(A[1:, :], 1/(C*h**2))
    np.fill_diagonal(A[:, 1:], 1/(C*h**2))
    A[0, -1] = 0
    A[-1, 0] = 0
    return A, xs

n = 1000
mat = matsetup_oscillator(n)
res = find_eigenpair(mat[0], np.random.rand(n), tol = 1e-20)
#gauss =np.linalg.eig(mat[0]) 
#min_eig = np.argmin(np.abs(gauss[0]))
#
#print(f"Error: {res[1] - gauss[0][min_eig]}")
print(f"Energy value: {res[1]/ev} eV") 
E_real = h_bar * omega * 0.5
print(f"Error: {round((res[1]/E_real - 1)*100, 2)}%")
y = u.positive_vector(u.normalize(res[0]))
x = mat[1]
#plt.plot(x, u.positive_vector(u.normalize (gauss[1][:, min_eig])))
plt.plot(x, y)
plt.grid()

plt.show()

