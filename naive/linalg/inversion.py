import numpy as np
from .factorization import factorize_LU, retrieve_LU, factorize_QR, retrieve_QR

def inverse_D(D):
    m, n = D.shape
    assert m == n
    X = np.zeros_like(D)
    for i in range(n):
        X[i, i] = 1 / D[i, i]
    return X

def inverse_L(L):
    m, n = L.shape
    assert m == n
    X = np.copy(L)
    for i in range(n):
        for j in range(i):
            X[i ,j] = - X[j:i, j] @ X[i, j:i] / X[i, i]
        X[i, i] = 1 / X[i, i]
    return X

def inverse_U(U):
    return inverse_L(U.T).T

def inverse(A, method="LU"):
    # TODO: check singularity
    
    if method == "LU":
        L, U = retrieve_LU(factorize_LU(A))
        return inverse_U(U) @ inverse_L(L)
    elif method == "QR":
        Q, R = retrieve_QR(factorize_QR(A))
        return inverse_U(R) @ Q.T