import numpy as np
import time
from scipy.optimize import minimize_scalar
def find_eigenpair_constrained(A, x0, C, h, tol = 1e-20, n_max = 100000, verbose = False):
    # Rayleigh quotient definition
    f = lambda x: np.dot(x, A @ x)/np.dot(x, x)
    # Rayleigh quotient gradient
    g = lambda x: 2*(A @ x - f(x)*x)/np.dot(x, x)

    # Initial guess
    x = x0
    # Initial guess gradient
    grad = g(x)
    # Initial guess gradient norm, to be used for stopping criterion
    g0 = np.dot(grad, grad)
    # Initial search direction
    d = -grad
    r_new = -grad

    n = 0

    for i in range(n_max):
        r_old = r_new
        x = x + find_positive_root(A, x, d) * d
        grad = g(x)
        
        # Convergence check, as suggested by (1)
        if n > 10 and np.dot(grad, grad) < tol*g0:
            break
        r_new = -grad

        b = compute_beta_PR(r_new, r_old)
        d = r_new + b * d
        n = i
        if verbose:
            print(f"iteration {i}")
    
    print(f"Done in {n} iterations")
    print(f"Smallest eigenvalue: {f(x)}") 
    return (x, f(x))

def mat_mult(A, x, n):
    y = np.zeros(n)
    for i in range(n):
        for j in range(n):
            y[i] = y[i] + A(i, j)*x[j]

def find_eigenpair_unalloc(A, x0, tol = 1e-20, n_max = 100000, verbose = False):
    # Rayleigh quotient definition
    f = lambda x, Ax, xx: np.dot(x, Ax)/xx
    # Rayleigh quotient gradient
    g = lambda x, Ax, xx: 2*(Ax - f(x, Ax, xx)*x)/xx

    # Initial guess
    x = x0
    n = x.shape[0]
    # Initial Ax, xx, needed for performance
    Ax = mat_mult(A, x, n)
    xx = np.dot(x, x)
    # Initial guess gradient
    grad = g(x, Ax, xx)
    # Initial guess gradient norm, to be used for stopping criterion
    g0 = np.dot(grad, grad)
    # Initial search direction
    d = -grad
    r_new = -grad

    n = 0

    for i in range(n_max):
        r_old = r_new
        x = x + find_positive_root(A, x, d, Ax, xx) * d
        Ax = mat_mult(A, x, n)
        xx = np.dot(x, x)
        grad = g(x, Ax, xx)
        
        # Convergence check, as suggested by (1)
        if n > 10 and np.dot(grad, grad) < tol*g0:
            break
        r_new = -grad

        b = compute_beta_PR(r_new, r_old)
        d = r_new + b * d
        n = i
        if verbose:
            print(f"iteration {i}")
    l = f(x, Ax, xx)
    print("Tolerance reached: ", np.dot(grad, grad)/g0)
    print(f"Done in {n} iterations")
    print(f"Smallest eigenvalue: {l}") 
    return (x, l)
def find_eigenpair(A, x0, tol = 1e-20, n_max = 100000, verbose = False):
    # Rayleigh quotient definition
    f = lambda x, Ax, xx: np.dot(x, Ax)/xx
    # Rayleigh quotient gradient
    g = lambda x, Ax, xx: 2*(Ax - f(x, Ax, xx)*x)/xx

    # Initial guess
    x = x0
    # Initial Ax, xx, needed for performance
    Ax = A @ x
    xx = np.dot(x, x)
    # Initial guess gradient
    grad = g(x, Ax, xx)
    # Initial guess gradient norm, to be used for stopping criterion
    g0 = np.dot(grad, grad)
    # Initial search direction
    d = -grad
    r_new = -grad

    n = 0

    for i in range(n_max):
        r_old = r_new
        x = x + find_positive_root(A, x, d, Ax, xx) * d
        Ax = A @ x
        xx = np.dot(x, x)
        grad = g(x, Ax, xx)
        
        # Convergence check, as suggested by (1)
        if n > 10 and np.dot(grad, grad) < tol*g0:
            break
        r_new = -grad

        b = compute_beta_PR(r_new, r_old)
        d = r_new + b * d
        n = i
        if verbose:
            print(f"iteration {i}")
    l = f(x, Ax, xx)
    print("Tolerance reached: ", np.dot(grad, grad)/g0)
    print(f"Done in {n} iterations")
    print(f"Smallest eigenvalue: {l}") 
    return (x, l)


def find_positive_root(A, x, p, Ax, xx):

    Ap = A @ p
    xp = np.dot(x, p)
    pp = np.dot(p, p)
    pAp = np.dot(p, Ap)
    xAp = np.dot(x, Ap)
    xAx = np.dot(x, Ax)
   
    # p search direction, A matrix, x point
    a = (pAp) * (xp) - (xAp) * (pp)
    b = (pAp) * (xx) - (xAx) * (pp)
    c = (xAp) * (xx) - (xAx) * (xp)
    
    delta = b**2 - 4*a*c
    
    if delta < 0:
        return 0  
    
    sqrt_delta = np.sqrt(delta)
    root1 = (-b + sqrt_delta) / (2 * a)
    root2 = (-b - sqrt_delta) / (2 * a)
    
    # Return the positive root
    roots = [r for r in (root1, root2) if r > 0]
    return np.min(roots) if roots else 0

def compute_beta_FR(r_new, r_old):
    return np.dot(r_new, r_new)/np.dot(r_old, r_old)
def compute_beta_PR(r_new, r_old):
    return max(np.dot(r_new, r_new-r_old)/np.dot(r_old, r_old), 0)

def test():
    n = 1000
    A = np.random.rand(n, n)

    # Rendiamola simmetrica
    A_sym = (A + A.T) / 2 
    A_sym = A_sym
    t = time.perf_counter()
    res = np.linalg.eig(A_sym)
    t = time.perf_counter() - t
    print(t)
    print(res[1][:, np.argmin(res[0])])
    #print(res[1][:, 0])
    t = time.perf_counter()
    cgm = find_eigenpair(A_sym, np.random.rand(n))
    print(cgm[0]/np.dot(cgm[0], cgm[0])**0.5)
    t = time.perf_counter() - t
    print(t)
    print(f"Error: {np.min(res[0]) - cgm[1]}")
    lob = lobpcg(A_sym, np.linalg.qr(np.random.rand(n, 2))[0])
    print(f"Error lobpcg: {np.min(lob[0]) - np.min(res[0])}")
if __name__ == "__main__":
    print("Running eigensolvers tests")
    test()