import numpy as np
from scipy.sparse.linalg import cg, eigsh

def rayleigh_quotient(A, x):
    """ Calcola il quoziente di Rayleigh R(A, x) """
    return (x.T @ A @ x) / (x.T @ x)

def inverse_power_iteration(A, x, sigma, tol=1e-6, max_iter=100):
    """ Metodo della potenza inversa smorzata con gradiente coniugato """
    I = np.eye(A.shape[0])
    for _ in range(max_iter):
        # Risolvi (A - sigma I)y = x usando il gradiente coniugato
        y, _ = cg(A - sigma * I, x, rtol=tol)
        y /= np.linalg.norm(y)  # Normalizza
        if np.linalg.norm(A @ y - sigma * y) < tol:
            break
        x = y
    return y

def gcg_eigenvalues(A, num_eigenvalues=2, tol=1e-6, max_iter=100):
    """ Algoritmo GCG per trovare i primi due autovalori più piccoli """
    n = A.shape[0]
    
    # 1. Inizializza un set di vettori casuali
    X = np.random.rand(n, num_eigenvalues)
    X, _ = np.linalg.qr(X)  # Ortogonalizza

    # 2. Itera fino alla convergenza
    for _ in range(max_iter):
        W = np.zeros_like(X)
        
        # 2.1. Passo della potenza inversa smorzata per generare W
        for i in range(num_eigenvalues):
            W[:, i] = inverse_power_iteration(A, X[:, i], sigma=0.0, tol=tol)

        # 2.2. Ortogonalizzazione completa
        Q, _ = np.linalg.qr(np.hstack([X, W]))

        # 2.3. Proiezione di Rayleigh-Ritz
        A_reduced = Q.T @ A @ Q  # Proietta A nel sottospazio ridotto
        eigenvalues, eigenvectors = np.linalg.eigh(A_reduced)

        # 2.4. Ricostruzione degli autovettori nel dominio originale
        X_new = Q @ eigenvectors[:, :num_eigenvalues]

        # 2.5. Controllo della convergenza
        residuals = np.linalg.norm(A @ X_new - X_new @ np.diag(eigenvalues[:num_eigenvalues]), axis=0)
        if np.all(residuals < tol):
            break
        
        X = X_new  # Aggiorna i vettori
    
    return eigenvalues[:num_eigenvalues], X