import numpy as np
from .inversion import inverse_D, inverse_L
from .solution import get_DLU

def solve_iterative(A, b, method="Jacobi", x0=None, max_step=100000, tol=0.0001, omega=None, verbose=False):
    # A = D-L-U
    # Jacobi: x' = D^{-1}(L+U) x + D^{-1} b
    # GS & SOR: x' = (D-L)^{-1}U x + (D-L)^{-1} b

    m, n = A.shape
    D, L, U = get_DLU(A)
    L, U = -L, -U
    
    if method == "GS" or method == "SOR":
        DL_inv = inverse_L(D-L)
        M = DL_inv @ U
        g = DL_inv @ b
    elif method == "Jacobi":
        D_inv = inverse_D(D)
        M = D_inv @ (L + U)
        g = D_inv @ b
    else:
        raise NotImplementedError

    if method == "GS" or method == "Jacobi":
        omega = 1
    elif method == "SOR":
        if omega is None:
            D_inv = inverse_D(D)
            B = D_inv @ (L + U)
            
            #FIXME: used numpy builtin eigvals
            rou = np.max(np.abs(np.linalg.eigvals(B)))
            omega = 2 / (1 + np.sqrt(1 - rou**2))
            
            if verbose:
                print(f"Estimated omega for SOR: {omega}")

    assert 0 < omega < 2

    # TODO: determine convergency

    # iteration
    step = 0
    x = x0 if x0 is not None else np.zeros(n)

    while True:
        old_x = np.copy(x)
        x = x + omega * (M @ x + g - x)
        # manual limit
        step += 1
        if max_step is not None and step >= max_step: 
            if verbose:
                print(f"Early stop at step {step}")
            break
        # tolerance limit
        if np.linalg.norm(A @ x - b, ord=1) < tol:
            if verbose:
                print(f"Covengent at step {step} under tolerance {tol}")
            break
    
    return x



    
