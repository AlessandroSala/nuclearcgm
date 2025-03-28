import numpy as np
import matplotlib.pyplot as plt
from eigen import find_eigenpair
from nuc_constants import *
import util as u
A = 16
a = 10
omega = 41/h_bar/A**(1/3)

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
res = find_eigenpair(mat[0], np.random.rand(n), tol = 1e-26, n_max=5000)
#gauss =np.linalg.eig(mat[0]) 
#min_eig = np.argmin(np.abs(gauss[0]))
#
#print(f"Error: {res[1] - gauss[0][min_eig]}")
print(f"GS Energy value: {res[1]} MeV") 
E_real = h_bar * omega * 0.5
print(f"Error GS: {round((res[1]/E_real - 1)*100, 2)}%")
y = u.positive_vector(u.normalize(res[0]))
x = mat[1]
#plt.plot(x, u.positive_vector(u.normalize (gauss[1][:, min_eig])))
#res_2 = find_eigenpair_constrained(mat[0], u.normalize(guess), u.normalize(res[0]), tol = 1e-22, c= 1000)
plt.plot(x, y**2, label="$|\\psi|^2$")
plt.xlabel("$x$ [fm]")
plt.legend()
#plt.plot(x, (u.normalize(res_2[0])))
plt.grid()
plt.show()

print(mat[0])

#print(f"Error ES: {round((res_2[1]/(h_bar*omega*1.5) - 1)*100, 2)}%")
#print(np.column_stack((res[0], guess)).shape)
#res_cgc = lobpcg(mat[0], np.column_stack((np.random.rand(n), np.random.rand(n))), tol=1e-30, maxiter=100000) 
#print(f"Error gcg GS: {round((res_cgc[0][0]/(h_bar*omega*0.5) - 1)*100, 2)}%")
#plt.show()

