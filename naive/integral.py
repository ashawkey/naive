import numpy as np
from . constants import *

def integral_mid(f, a, b):
    return f((a+b)/2) * (b-a)

def integral_trapezoid(f, a, b):
    return (f(a) + f(b)) * (b - a) / 2

def integral_simpson(f, a, b):
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))

def integral_mid_composite(f, a, b, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        res += dx * f(i + dx/2)
    return res

def integral_trapezoid_composite(f, a, b, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        res += dx * (f(i) + f(i + dx)) / 2
    return res

def integral_simpson_composite(f, a, b, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        res += dx * (f(i) + 4 * f(i + dx/2) + f(i + dx)) / 6
    return res

def integral(f, a, b, method='simpson', dx=DX):
    if method == 'mid':
        return integral_mid_composite(f, a, b, dx)
    elif method == 'trapezoid':
        return integral_trapezoid_composite(f, a, b, dx)
    elif method == 'simpson':
        return integral_simpson_composite(f, a, b, dx)
    else:
        raise NotImplementedError

def double_integral_mid_composite(f, a, b, fa, fb, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        for j in np.arange(fa(i), fb(i), dx):
            res += dx * dx * f(i + dx/2, j + dx/2)
    return res

def double_integral_trapezoid_composite(f, a, b, fa, fb, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        for j in np.arange(fa(i), fb(i), dx):
            res += dx * dx * (f(i, j) + f(i+dx, j) + f(i, j+dx) + f(i+dx, j+dx)) / 4
    return res

def double_integral_simpson_composite(f, a, b, fa, fb, dx=DX):
    res = 0
    for i in np.arange(a, b, dx):
        for j in np.arange(fa(i), fb(i), dx):
            res += dx * dx * (f(i, j) + f(i+dx, j) + f(i, j+dx) + f(i+dx, j+dx) + 
                              4 * (f(i, j+dx/2) + f(i+dx, j+dx/2) + f(i+dx/2, j) + f(i+dx/2, j+dx)) + 
                              16 * f(i+dx/2, j+dx/2)) / 36
    return res


def double_integral(f, a, b, fa, fb, method='simpson', dx=DX):
    if method == 'mid':
        return double_integral_mid_composite(f, a, b, fa, fb, dx)
    elif method == 'trapezoid':
        return double_integral_trapezoid_composite(f, a, b, fa, fb, dx)
    elif method == 'simpson':
        return double_integral_simpson_composite(f, a, b, fa, fb, dx)
    else:
        raise NotImplementedError