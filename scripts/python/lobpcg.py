import numpy as np

def rayleigh_ritz(S, A, B=None):
    """ Implementa la procedura di Rayleigh-Ritz """
    if B is None:
        B = np.eye(S.shape[0])

    M = S.T @ B @ S
    D = np.diag(np.diag(M)**(-0.5))
    R = np.linalg.cholesky(D @ M @ D)
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(R.T) @ D @ (S.T @ A @ S) @ D @ np.linalg.inv(R))  
    
    return eigvecs, np.diag(eigvals)

def lobpcg_simple(A, X0, B=None, max_iter=1000, tol=1e-25):
    """ Implementazione semplificata dell'algoritmo LOBPCG """
    if B is None:
        B = np.eye(A.shape[0])

    X = X0.copy()
    C, Theta = rayleigh_ritz(X, A, B)
    X = X @ C
    R = A @ X - B @ X @ Theta
    P = None

    for i in range(max_iter):
        if np.linalg.norm(R) < tol:
            print("Convergenza raggiunta in ", i, " iterazioni")
            break

        W = R  # Risolve B W = R
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

# Test
def test_lobpcg():
    np.random.seed(42)
    n = 100
    eigenvalues = np.linspace(1, 100, n)  # Autovalori ben separati tra 1 e 100
    Q, _ = np.linalg.qr(np.random.rand(n, n))  # Matrice ortogonale casuale
    A = Q @ np.diag(eigenvalues) @ Q.T
    X0 = np.random.rand(n, 2)  # Generiamo due vettori casuali
    X0, _ = np.linalg.qr(X0)   # Ortonormalizzazione con QR


    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eigh(A)

    eigenvalues_lobpcg, eigenvectors_lobpcg = lobpcg_simple(A, eigenvectors_numpy[:, :2])

    print("LOBPCG autovalori:", np.sort(np.diag(eigenvalues_lobpcg)))
    print("NumPy autovalori:", np.sort(eigenvalues_numpy[:2]))

if __name__ == "__main__":
    test_lobpcg()
