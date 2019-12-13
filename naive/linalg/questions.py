import numpy as np
import scipy

def question_1_1(l=84):
    A = np.zeros((l, l))
    b = np.zeros(l)

    for i in range(l):
        A[i, i] = 6
        if i != 0: A[i, i-1] = 8
        if i != l-1: A[i, i+1] = 1

        if i == 0: b[i] = 7
        elif i == l-1: b[i] = 14
        else: b[i] = 15

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    x = np.ones(l)
    return A, b, x

def question_1_2_1(l=100):
    A = np.zeros((l, l))
    for i in range(l):
        A[i, i] = 10
        if i != 0: A[i, i-1] = 1
        if i != l-1: A[i, i+1] = 1
    b = np.random.rand(100)
    x = scipy.linalg.solve(A, b) # scipy as the ground truth
    return A, b, x
    

def question_1_2_2(l=40):
    A = np.zeros((l, l))
    b = np.zeros(l)
    for i in range(l):
        for j in range(l):
            A[i, j] = 1 / (i + j + 1);
            b[i] += A[i, j]
    x = np.ones(l)
    return A, b, x

def question_4_1(n=100, a=1/2, epsilon=1):
    def get_res(x):
        return (1-a)/(1-np.e**(-1/epsilon))*(1-np.e**(-x/epsilon))+a*x
    h = 1 / n
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    x = np.zeros(n-1)
    for i in range(n-1):
        b[i] = a * h**2
        if i == n-2: b[i] -= epsilon + h # x[-1] = 0, x[n] = 1
        x[i] = get_res((i+1)*h)
        for j in range(n):
            if i>0: A[i, i-1] = epsilon
            A[i, i] = - (2 * epsilon + h)
            if i<n-2: A[i, i+1] = epsilon + h
    return A, b, x
    
def question_6_1(a=[3, -5, 1]):
    n = len(a)
    A = np.zeros((n,n))
    A[n-1, 0] = 1
    A[n-1, n-1] = -a[n-1]
    for i in range(1, n):
        A[0, i] = -a[i-1]
        if i > 1: A[i-1, i] = 1
    return A

