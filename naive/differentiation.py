import numpy as np
from . constants import *

"""
This is bad. I use continuous function instead of discrete array to represent a function.
The differentiation cannot be automatically extended to high order. Have to hard-code the formula...
ref: https://en.wikipedia.org/wiki/Numerical_differentiation
"""

def diff_forward(f, x, dx=DX):
    return (f(x+dx) - f(x)) / dx

def diff_backward(f, x, dx=DX):
    return (f(x) - f(x-dx)) / dx

def diff_central(f, x, dx=DX):
    return (f(x+dx) - f(x-dx)) / (2 * dx)

def diff2_central(f, x, dx=DX):
    return (f(x+dx) - 2 * f(x) + f(x-dx)) / dx**2

def diff(f, x, order=1, method='central', dx=DX):
    '''
    uni-variable 1/2 order differentiation.
    '''
    if method == 'central':
        if order == 1:
            return diff_central(f, x, dx)
        elif order == 2:
            return diff2_central(f, x, dx)
        else:
            raise NotImplementedError
    elif method == 'forward':
        if order == 1:
            return diff_forward(f, x ,dx)
        else:
            raise NotImplementedError
    elif method == 'backward':
        if order == 1:
            return diff_backward(f, x ,dx)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

### Partial Dirivatives

def diff_partial_central(f, x, index, dx=DX):
    '''
    1-order partial differentiation.
    f: multi-variable function, receiving a list of vars.
        eg. def f(x): return x[0] + x[1]
    x: list/array of vars.
    '''
    dim = len(x)
    dx_ = np.array([dx if i == index else 0 for i in range(dim)])
    return (f(x+dx_) - f(x-dx_)) / (2 * dx)

def diff2_partial_central(f, x, index0, index1, dx=DX):
    '''
    2-order partial.
    ref: https://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h07/undervisningsmateriale/kap7.pdf
    '''
    dim = len(x)
    if index0 == index1:
        # specially for second-order
        dx_ = np.array([dx if i == index0 else 0 for i in range(dim)])
        return (f(x+dx_) - 2 * f(x) + f(x-dx_)) / dx**2
    else:
        # mixed
        dx_0 = np.array([dx if i == index0 else 0 for i in range(dim)])
        dx_1 = np.array([dx if i == index1 else 0 for i in range(dim)])
        return (f(x+dx_0+dx_1) - f(x+dx_0-dx_1) - f(x-dx_0+dx_1) + f(x-dx_0-dx_1)) / (4 * dx**2)

def gradient(f, x, dx=DX):
    dim = len(x)
    return np.array([diff_partial_central(f, x, i, dx) for i in range(dim)])

def hessian(f, x, dx=DX):
    dim = len(x)
    res = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            res[i, j] = diff2_partial_central(f, x, i, j, dx)
            if i != j:
                res[j, i] = res[i, j]
    return res

