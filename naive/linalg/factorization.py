import numpy as np

'''
numpy vector multiplication

for clearity, we assert a is column-vector, though numpy will not distinguish row/column vector.

* vector @ vector
    * return scalar: a.T @ b
        a @ b
        a @ b.T
        a.T @ b
        a.T @ b.T
        (in fact a == a.T)
    * return matrix: a @ b.T
        a[:,None] @ b[None]
        a[:,None] @ b[None,:]
        a.reshape(-1,1) @ b.reshape(1,-1)

* vector @ matrix
    * left mult: a.T @ M
        a @ M
        a.T @ M
    * right mult: M @ a
        M @ a

* others
    * return scalar: a.T @ M @ b
        a @ M @ b
'''

def factorize_Cholesky(A):
    # Cholesky decomposition, return L
    n, n = A.shape
    L = np.copy(A)
    for i in range(n):
        L[i,i] = np.sqrt(np.abs(L[i,i]))
        L[i+1:n, i] /= L[i,i]
        for j in range(i+1, n):
            L[j:n, j] -= L[j:n, i] * L[j, i]
    return L

def factorize_LDL(A):
    # LDL' decomposition, LD compressed in one matrix.
    n, n = A.shape
    L = np.copy(A)
    v = np.zeros(n)
    for i in range(n):
        for j in range(i):
            v[j] = L[i, j] * L[j, j]
        L[i, i] -= L[i, 0:i] @ v[0:i]
        L[i+1:n, i] -= L[i+1:n, 0:i] @ v[0:i]
        L[i+1:n, i] /= L[i, i]
    return L

def factorize_LU(A):
    # LU decomposition, LU compressed in one matrix
    m, n = A.shape
    assert m == n
    X = np.copy(A)
    for i in range(n-1):
        X[i+1:n, i] = X[i+1:n, i] / X[i, i]
        X[i+1:n, i+1:n] -= X[i+1:n, i][:,np.newaxis] @ X[i, i+1:n][np.newaxis,:]
    return X

def retrieve_LU(LU):
    m, n = LU.shape
    assert m == n
    L = np.zeros_like(LU)
    U = np.zeros_like(LU)
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i, j] = 1
                U[i, j] = LU[i, j]
            elif i > j:
                L[i, j] = LU[i, j]
            else:
                U[i, j] = LU[i, j]
    return L, U

def factorize_PLU(A):
    # PLU decomposition, LU compressed in one matrix
    n, n = A.shape
    X = np.copy(A)
    P = np.eye(n)
    for i in range(n-1):
        idx = np.argmax(np.abs(X[i:, i]))
        X[[i, i+idx]] = X[[i+idx, i]]
        P[[i, i+idx]] = P[[i+idx, i]]
        X[i+1:n, i] /= X[i, i]
        X[i+1:n, i+1:n] -= X[i+1:n, i][:,np.newaxis] @ X[i, i+1:n][np.newaxis,:]
    return P, X

def transform_Householder(x):
    # H @ x = [1, 0, 0, ...]
    # H = I - beta @ v @ v.T
    n = x.shape[0]
    v = np.copy(x)
    v = v / np.max(np.abs(v))
    sigma = v[1:].T @ v[1:]
    
    if sigma == 0:
        beta = 0
    else:
        alpha = np.sqrt(v[0]**2 + sigma)
        if v[0] <= 0:
            v[0] = v[0] - alpha
        else:
            v[0] = - sigma / (v[0] + alpha)
        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]
    return beta, v

def retrieve_H(beta, v, length=None):
    n = v.shape[0]
    if length is None:
        length = n
    H = np.eye(length)
    v = np.hstack([[0]*(length-n), v])
    H -= beta * v.reshape(-1,1) @ v.reshape(1,-1)
    return H

def factorize_QR(A):
    # save QR in X, d
    m, n = A.shape
    d = np.zeros(n)
    X = np.copy(A)
    for j in range(n):
        if j < m:
            beta, v = transform_Householder(X[j:, j])
            X[j:, j:] = retrieve_H(beta, v) @ X[j:, j:]
            d[j] = beta
            X[j+1:, j] = v[1:]
    return X, d

def retrieve_QR(X, d):
    m, n = X.shape
    Q = np.eye(m)
    R = np.zeros_like(X)
    for j in range(n):
        beta = d[j]
        v = np.hstack([[1], X[j+1:, j]])
        H = retrieve_H(beta, v, m)
        Q = Q @ H
        R[j, j:] = X[j, j:]
    return Q, R