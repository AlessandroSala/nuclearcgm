import numpy as np
import time
from scipy.optimize import minimize_scalar
def find_eigenpair(A, x0, tol = 1e-20, n_max = 100000, verbose = False):
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

def find_eigenpair_constrained(A, x0, K, tol = 1e-20, c = 5):
    n = A.shape[0]
    B = np.eye(n)
    C1 = 0
    C2 = 0
    W = np.zeros(n)
    P = np.zeros(n)
    X = x0
    W = np.linalg.inv(K) @ (A @ X - B @ X @ (X.T @ A @ X))

    # Rayleigh quotient gradient
    g = lambda x: 2*(A @ x - f(x)*x)/np.dot(x, x) + c*np.dot(w, x)*w
    # Initial guess
    x = x0
    # Initial guess gradient
    grad = g(x)
    # Initial guess gradient norm, to be used for stopping criterion
    g0 = np.dot(grad, grad)
    # Initial search direction
    d = -grad
    r_new = -grad


    for i in range(2000):
        r_old = r_new
        x = x + minimize_scalar(lambda a: f((x) + a * d)).x * d
        grad = g(x)
        
        # Convergence check, as suggested by (1)
        if n > 10 and np.dot(grad, grad) < tol*g0:
            break
        r_new = -grad

        b = compute_beta_PR(r_new, r_old)
        d = r_new + b * d
#    x = (np.identity(n)-w @ w.T)@x
    print(f"Smallest eigenvalue: {f(x)}") 
    return (x, f(x))
def lobpcg(A, X, Y=None, T=None, tol=1e-6, max_iter=100):
    """
    Implementazione dell'algoritmo LOBPCG basato sulla descrizione della sezione 2.2 del documento BLOPEX.
    
    Parametri:
    - A: np.ndarray, matrice simmetrica (n x n)
    - X: np.ndarray, m vettori iniziali (n x m)
    - Y: np.ndarray, l vettori di vincolo opzionali (n x l), default=None
    - T: funzione di precondizionamento, opzionale, default=None
    - tol: float, tolleranza per la convergenza
    - max_iter: int, numero massimo di iterazioni
    
    Ritorna:
    - Lambda: np.ndarray, primi m autovalori
    - X: np.ndarray, autovettori associati
    """
    n, m = X.shape
    
    # Allocazione memoria
    W = np.zeros_like(X)
    P = np.zeros_like(X)
    AX = np.zeros_like(X)
    BX = np.zeros_like(X)
    
    # Se ci sono vincoli Y, li applichiamo a X
    if Y is not None:
        BY = A @ Y
        X -= Y @ np.linalg.solve(Y.T @ BY, BY.T @ X)
    
    # B-ortonormalizzazione iniziale di X
    BX = A @ X
    R = np.linalg.cholesky(X.T @ BX)
    X = X @ np.linalg.inv(R)
    BX = BX @ np.linalg.inv(R)
    AX = A @ X
    
    # Calcolo iniziale degli autovalori tramite Rayleigh-Ritz
    B = X.T @ AX
    eigvals, eigvecs = np.linalg.eigh(B)
    X = X @ eigvecs
    AX = AX @ eigvecs
    BX = BX @ eigvecs
    
    for k in range(max_iter):
        # Residuo: W = AX - BX * Lambda
        W = AX - BX @ np.diag(eigvals)
        
        # Controllo convergenza
        if np.linalg.norm(W, ord='fro') < tol:
            break
        
        # Precondizionamento
        if T is not None:
            W = T(W)
        
        # B-ortonormalizzazione di W
        BW = A @ W
        R = np.linalg.cholesky(W.T @ BW)
        W = W @ np.linalg.inv(R)
        BW = BW @ np.linalg.inv(R)
        AW = A @ W
        
        # Costruzione della base per Rayleigh-Ritz
        G_A = np.block([[eigvals, X.T @ AW], [AW.T @ X, W.T @ AW]])
        G_B = np.block([[np.eye(m), X.T @ BW], [BW.T @ X, np.eye(m)]])
        
        # Risoluzione problema agli autovalori generalizzato
        eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(G_B, G_A))
        
        # Aggiornamento autovettori
        CX, CW = eigvecs[:m, :], eigvecs[m:, :]
        P = W @ CW + P @ CX
        X = X @ CX + P
        AX = AX @ CX + AW @ CW
        BX = BX @ CX + BW @ CW
    
    return eigvals, X



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