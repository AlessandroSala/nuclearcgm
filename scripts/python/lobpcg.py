import numpy as np
from scipy.sparse.linalg import lobpcg
import numpy as np
from eigen import find_eigenpair, g, find_eigenpair_rev
import util
import math
def rayleigh_ritz(S, A, B=None):
    """ Implementa la procedura di Rayleigh-Ritz """
    if B is None:
        B = np.eye(S.shape[0])

    M = S.T @ B @ S
    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(M) @ (S.T @ A @ S))
    
    return eigvecs, np.diag(eigvals)

def lobpcg_simple(A, X0, B=None, max_iter=100, tol=1e-6):
    """ Implementazione semplificata dell'algoritmo LOBPCG """
    if B is None:
        B = np.eye(A.shape[0])

    X = X0.copy()
    C, Theta = rayleigh_ritz(X, A, B)
    X = X @ C
    R = A @ X - B @ X @ Theta
    P = None

    for _ in range(max_iter):
        if np.linalg.norm(R) < tol:
            break

        W = np.linalg.solve(B, R)  # Risolve B W = R
        if P is None:
            S = np.hstack((X, W))
        else:
            S = np.hstack((X, W, P))

        C, Theta = rayleigh_ritz(S, A, B)
        k = X.shape[1]
        X_new = S @ C[:, :k]  # Prendiamo solo i primi k autovettori
        Theta = Theta[:k, :k]  # Prendiamo solo i primi k autovalori
        R = A @ X_new - B @ X_new @ Theta
        P = S[:, k:] @ C[k:, :k]  # Proiezione sui nuovi vettori
        X = X_new

    return Theta, X

def acgm(A, X0, tol = 1e-6, max_iter = 100):
    w = X0.copy()
    k = X0.shape[1]
    eigenvalues = np.zeros(k)
    diag_eigenvalues = np.zeros(k)

    for iteration in range(max_iter):
        for i in range(k):
            Q = proj_mat(w[:, :i])
            pair = find_eigenpair_rev(Q @ A @ Q, w[:, i], tol = tol, max_iter = max_iter)
            if(math.isnan(pair[1] ) ):
                print("NaN")
                return w, eigenvalues, diag_eigenvalues

            w[:, i] = pair[0]
            eigenvalues[i] = pair[1]
        M = w.T @ A @ w
        res = np.linalg.eig(M)
        idx = res[0].argsort()
        res = res[0][idx], res[1][:, idx]
        diag_eigenvalues = res[0]
        w_copy = w.copy()
        for i in range(k):
            s = 0
            for j in range(k):
                s += res[1][j, i] * w_copy[:, j]
            w[:, i] = s 
        
    return w, eigenvalues, diag_eigenvalues


def proj_mat(W):
    return np.eye(W.shape[0]) - W @ W.T
# Test
def test_lobpcg():
    #np.random.seed(43)
    n = 100
    k = 6
    A = np.random.rand(n, n)
    A = A + A.T
    X0 = np.random.rand(n, k)
    X0, _ = np.linalg.qr(X0)
    


    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eigh(A)

    eigenvalues_lobpcg, eigenvectors_lobpcg = lobpcg_simple(A, eigenvectors_numpy[:, :k])
    eigenvalues_lobpcg_scipy, eigenvectors_lobpcg_scipy = lobpcg(A, eigenvectors_numpy[:, :k], None, np.diag(np.diag(A)) , maxiter=1000, tol=1e-10)
    eigenvectors_acgm, eigenvalues_acgm, diag_eigenvalues_acgm = acgm(A, X0)

    #print("LOBPCG mio autovalori:", np.sort(np.diag(eigenvalues_lobpcg)))
    #print("LOBPCG scipy autovalori:", np.sort(eigenvalues_lobpcg_scipy))
    print("accelerated cgm ", np.sort(eigenvalues_acgm))
    print("accelerated cgm diagonal ", np.sort(diag_eigenvalues_acgm))
    #print("LOBPCG mio autovalori:", np.sort(eigenvalues_lobpcg_scipy))
    print("NumPy autovalori:", np.sort(eigenvalues_numpy[:k]))

if __name__ == "__main__":
    test_lobpcg()
