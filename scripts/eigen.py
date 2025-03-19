import numpy as np
import time

def find_eigenpair(A, x0, tol = 1e-20):
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

    for i in range(1000000):
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
    
    print(f"Done in {n} iterations")
    print(f"Smallest eigenvalue: {f(x)}") 
    return (x, f(x))

def find_positive_root(A, x, p):
   
    # p search direction, A matrix, x point
    a = (p.T @ A @ p) * (x.T @ p) - (x.T @ A @ p) * (p.T @ p)
    b = (p.T @ A @ p) * (x.T @ x) - (x.T @ A @ x) * (p.T @ p)
    c = (x.T @ A @ p) * (x.T @ x) - (x.T @ A @ x) * (x.T @ p)
    
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
    n = 10
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

#test()