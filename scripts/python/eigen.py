import numpy as np
import time
from scipy.optimize import minimize_scalar

# Rayleigh quotient definition
def f (x, Ax, xx):
    return x.T @ Ax / xx

# Rayleigh quotient gradient
def g (x, Ax, xx):
    return 2*(Ax - f(x, Ax, xx)*x)/ xx

def g_norm(x, Ax):
    return (Ax - f(x, Ax, x.T @ x)*x)

def find_eigenpair_rev(A, x0, tol = 1e-6, max_iter = 100, verbose = False):
    x = x0.copy() / np.linalg.norm(x0)
    Ax = A @ x
    f = lambda x, Ax: x.T @ Ax
    g = lambda x, Ax: Ax - f(x, Ax)*x
    l = f(x, Ax)
    g0 = g(x, Ax)
    p = g0
    z = A@p
    for i in range(max_iter):
        a = z.T @ x
        b = z.T @ p
        c = x.T @ p
        d = p.T @ p
        delta = (l*d - b)**2 - 4*(b*c - a*d)*(a-l*c)
        alfa = (l*d - b + np.sqrt(delta) )/(2*(b*c - a*d))
        gamma = np.sqrt(1+2*c*alfa+d*alfa**2)
        l = (l+a*alfa)/(1+c*alfa) # new
        x = (x+alfa*p)/gamma
        Ax = (Ax+alfa*z)/gamma
        grad = Ax - l*x
        if np.linalg.norm(grad) < tol*l:
            print("Done in ", i, " iterations")
            break
        beta = -(grad.T @ z)/(b)
        p = grad + beta*p
        z = A@p
    #print(f"Done in {max_iter} iterations")
    return x, l


def find_eigenpair(A, x0, tol = 1e-20, n_max = 100000, verbose = False):
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
        #if verbose:
            #print(f"iteration {i}")
    l = f(x, Ax, xx)
    if verbose:
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
    n = 100
    A = np.random.rand(n, n)

    # Rendiamola simmetrica
    A_sym = (A + A.T) / 2 
    t = time.perf_counter()
    res = np.linalg.eig(A_sym)
    t = time.perf_counter() - t
    print(t)
    t = time.perf_counter()
    cgm = find_eigenpair_rev(A_sym, np.random.rand(n), tol=1e-20, max_iter=10000, verbose=True)
    t = time.perf_counter() - t
    print(t)
    print(f"Error: {round(100*(np.min(res[0]) - cgm[1])/np.min(res[0]), 2)}%")
if __name__ == "__main__":
    print("Running eigensolvers tests")
    test()