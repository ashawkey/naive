import numpy as np

from .solution import solve
from .factorization import transform_Householder, retrieve_H

def power_method(A, u0=None, displace=0, max_step=10000, tolerance=1e-8, verbose=False, inverse=False, normalize=True):
    # power_method: return largest modulus eigenvalue and corresponding eigenvector
    # inverse = True: return smallest eig instead.
    m, n = A.shape
    assert m == n

    if u0 is None:
        u0 = np.ones(n)
    if displace != 0:
        A = A - displace * np.eye(m)

    u = u0
    step = 0
    while True:
        old_u = np.copy(u)

        if not inverse:
            y = A @ u
        else:
            # TODO: since A is fixed, we can save the LU decomposition and reuse to reduce computation.
            y = solve(A, u, method="PLU")

        mu = u[np.argmax(np.abs(u))]
        u = y / mu
    
        step += 1
        if step >= max_step:
            if verbose:
                print(f"Early stop at step {step}")
            break
        
        if np.linalg.norm(old_u-u, ord=1) < tolerance:
            if verbose:
                print(f"Covengent at step {step} under tolerance {tolerance}")
            break
    
    if inverse:
        mu = 1 / mu
    
    if displace != 0:
        mu += displace
    
    if normalize:
        u = u / np.linalg.norm(u, ord=np.inf)
    
    return mu, u


def eig_power(A):
    # return all eigenvalues and eigenvectors
    m, n = A.shape
    assert m == n
    
    mus = []
    us = []

    X = np.copy(A)
    for i in range(n):
        mu, u = power_method(X)

        mus.append(mu)
        us.append(u)

        H = retrieve_H(*transform_Householder(u))
        X = H @ X @ H
        X = X[1:, 1:]


    return mus, us

