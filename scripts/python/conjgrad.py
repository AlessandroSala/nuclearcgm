import numpy as np

def conjugate_gradient(A, b, x0, tol = 1e-10):
    
    # initial position from first guess
    x = x0
    # initial residual from first guess
    r = b - A @ x
    # initial search direction (no previous history)
    d = r

    for _ in range(b.size):
        r_norm_sq = np.dot(r, r)    
        
        # A d product saved for performance 
        Ad = A @ d                   

        # step size
        alpha = r_norm_sq / np.dot(d, Ad)

        # take the next step
        x = x + alpha * d

        # new residual, optimized thanks to r_new being orthogonal to explored subspace
        r_new = r - alpha * Ad

        # tolerance check, truncates the method
        if np.linalg.norm(r_new) < tol:
            break

        # GS conjugation to find the new search direction, orthogonal to previous ones
        beta = np.dot(r_new, r_new) / r_norm_sq
        d = r_new + beta * d
        r = r_new

    return x

def preconditioned_conjugate_gradient(A, b, x0, T, tol = 1e-10):
    T_inv = np.linalg.inv(T)
    x = x0
    r = b - A @ x
    d = T_inv @ r

    for _ in range(b.size):
        r_norm_sq = np.dot(r, r)    
        
        # A d product saved for performance 
        Ad = A @ d                   

        T_invr = T_inv @ r
        # step size
        alpha = np.dot(r, T_inv @ r)/ np.dot(d, Ad)

        # take the next step
        x = x + alpha * d

        # new residual, optimized thanks to r_new being orthogonal to explored subspace
        r_new = r - alpha * Ad

        T_invr_new = T_inv @ r_new

        # tolerance check, truncates the method
        if np.linalg.norm(r_new) < tol:
            break


        # GS conjugation to find the new search direction, orthogonal to previous ones
        beta = np.dot(r_new, T_invr_new) / np.dot(r_norm_sq, T_invr)
        d = T_invr_new + beta * d
        r = r_new

    return x

                   
