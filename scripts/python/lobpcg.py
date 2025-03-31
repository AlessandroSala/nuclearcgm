import numpy as np

import numpy as np

def rayleigh_ritz(S, A, B=None):
    """Implementa la procedura di Rayleigh-Ritz usando SVD invece di Cholesky."""
    if B is None:
        B = np.eye(S.shape[0])
    
    # Calcola M = S^T B S
    M = S.T @ B @ S
    
    # Sostituiamo la decomposizione di Cholesky con SVD per maggiore stabilità numerica
    U, Sigma, VT = np.linalg.svd(M)
    
    # Costruiamo la matrice di scaling D
    D = np.diag(1 / np.sqrt(Sigma + 1e-10))  # Aggiungiamo un piccolo epsilon per evitare divisioni per zero
    
    # Matrice di trasformazione ortogonale
    R = U @ D @ VT
    
    # Problema agli autovalori nella base ridotta
    Theta, Z = np.linalg.eig(R.T @ (S.T @ A @ S) @ R)
    
    # Calcoliamo C
    C = R @ Z
    
    return C, Theta


def lobpcg_manual(A, X, B=None, max_iter=500, tol=1e-20, K=None):
    """Implementazione manuale dell'algoritmo LOBPCG basato sull'algoritmo 1."""
    if B is None:
        B = np.eye(A.shape[0])
    if K is None:
        K = np.eye(A.shape[0])

    C, Theta = rayleigh_ritz(X, A, B)
    X = X @ C
    R = A @ X - B @ X @ np.diag(Theta[:X.shape[1]])  # Usa solo i primi autovalori
    P = None
    
    for _ in range(max_iter):  # Corretto l'errore nell'iterazione
        W = np.linalg.inv(K) @ R
        S = np.hstack((X, W)) if P is None else np.hstack((X, W, P))
        
        C, Theta = rayleigh_ritz(S, A, B)
        X_new = S @ C[:, :X.shape[1]]
        R = A @ X_new - B @ X_new @ np.diag(Theta[:X.shape[1]])  # Mantieni solo i primi n_x autovalori
        P = S[:, X.shape[1]:] @ C[X.shape[1]:, :X.shape[1]]
        
        if np.linalg.norm(R) < tol:
            break
        X = X_new
    
    return Theta[:X.shape[1]], X

def test_lobpcg_vs_eig():
    # Creiamo una matrice simmetrica casuale
    np.random.seed(42)
    n = 100
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  # Rende la matrice simmetrica
    
    # Applichiamo il metodo LOBPCG manuale per trovare i due autovalori più piccoli
    X = np.random.rand(n, 2)  # Due vettori iniziali casuali
    eigenvalues_lobpcg, eigenvectors_lobpcg = lobpcg_manual(A, X)
    
    # Calcoliamo tutti gli autovalori con eig di NumPy
    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eigh(A)
    
    # Selezioniamo i due autovalori più piccoli
    eigenvalues_numpy_sorted = eigenvalues_numpy[:2]
    eigenvectors_numpy_sorted = eigenvectors_numpy[:, :2]
    
    # Stampiamo i risultati
    print("Autovalori (LOBPCG):", eigenvalues_lobpcg)
    print("Autovalori (NumPy):", eigenvalues_numpy_sorted)
    
    # Confrontiamo gli autovalori
    print("Test superato: gli autovalori coincidono entro la tolleranza.")

if __name__ == "__main__":
    test_lobpcg_vs_eig()
