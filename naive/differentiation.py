import numpy as np
from naive.constants import *

def diff_forward(f, x, dx=DX):
    return (f(x+dx) - f(x)) / dx

def diff_backward(f, x, dx=DX):
    return (f(x) - f(x-dx)) / dx

def diff_central(f, x, dx=DX):
    return (f(x+dx) - f(x-dx)) / (2 * dx)

def diff2_central(f, x, dx=DX):
    return (f(x+dx) - 2 * f(x) + f(x-dx)) / dx**2

def diff(f, x, order=1, method='central', dx=DX):
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