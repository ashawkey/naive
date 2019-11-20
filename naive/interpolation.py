import numpy as np
import math

def divdiff(xs, ys):
    """
    return divided-difference f[xs[0], xs[1], ...]
    """
    assert len(xs) == len(ys)
    m = len(xs)
    res = 0
    for i in range(m):
        quot = 1
        for j in range(m):
            if j != i:
                quot *= (xs[i] - xs[j])
        res += ys[i] / quot
    return res

def divdiff_all(xs, ys):
    """
    return [[f[x0], f[x1], ...], [f[x0, x1], f[x1, x2], ...], ...]
    """
    assert len(xs) == len(ys)
    m = len(xs)
    res = [ys]
    for i in range(1, m):
        tmp = []
        for j in range(m - i):
            tmp.append((res[-1][j+1] - res[-1][j]) / (xs[j+i] - xs[j]))
        res.append(tmp)
    return res


def Newton(xs, ys):
    """
    return (len(xs)-1)-th Newton interpolation function
    """
    assert len(xs) == len(ys)
    m = len(xs)
    diffs = divdiff_all(xs, ys)

    def f(x):
        res = 0
        for i in range(m):
            g = diffs[i][0]
            for j in range(i):
                g *= (x - xs[j])
            res += g
        return res

    return f

def lagrange_bases(xs):
    m = len(xs)

    res = []

    def make_base(i):
        def f(x):
            res = 1
            for j in range(m):
                if j != i:
                    res *= (x - xs[j]) / (xs[i] - xs[j])
            return res
        return f

    for i in range(m):
        res.append(make_base(i))
    
    return res

def Lagrange(xs, ys):
    assert len(xs) == len(ys)
    m = len(xs)

    bases = lagrange_bases(xs)

    def f(x):
        res = 0
        for i in range(m):
            res += ys[i] * bases[i](x)
        return res

    return f

def piecewise_linear_bases(xs):
    m = len(xs)
    bases = []

    def make_base(x0, x1, x2):
        def f(x):
            if x0 <= x <= x1:
                return (x - x0) / (x1 - x0)
            elif x1 <= x <= x2:
                return (x - x2) / (x1 - x2)
            else:
                return 0
        return f

    for i in range(m):
        if i == 0:
            bases.append(make_base(xs[0], xs[0], xs[1]))
        elif i == m-1:
            bases.append(make_base(xs[m-2], xs[m-1], xs[m-1]))
        else:
            bases.append(make_base(xs[i-1], xs[i], xs[i+1]))
    
    return bases


def PiecewiseLinear(xs, ys):
    assert len(xs) == len(ys)
    m = len(xs)
    
    bases = piecewise_linear_bases(xs)

    def f(x):
        res = 0
        for i in range(m):
            res += ys[i] * bases[i](x)
        return res

    return f


def hermite_bases(xs):
    m = len(xs)

    def make_base(x0, x1, x2):
        def alpha(x):
            return (1 + 2 * (x - x1) / (x2 - x1)) * ((x - x2) / (x1 - x2))**2
        def tilde_alpha(x):
            return (1 + 2 * (x - x1) / (x0 - x1)) * ((x - x0) / (x1 - x0))**2
        def beta(x):
            return (x - x1) * ((x - x2) / (x1 - x2))**2
        def tilde_beta(x):
            return (x - x1) * ((x - x0) / (x1 - x0))**2        
        
        def f(x):
            if x0 <= x <= x1:
                return tilde_alpha(x)
            elif x1 <= x <= x2:
                return alpha(x)
            else:
                return 0
        
        def g(x):
            if x0 <= x <= x1:
                return tilde_beta(x)
            elif x1 <= x <= x2:
                return beta(x)
            else:
                return 0         

        return f, g

    bases0 = []
    bases1 = []

    for i in range(m):
        if i == 0:
            f, g = make_base(xs[0], xs[0], xs[1])
        elif i == m-1:
            f, g = make_base(xs[m-2], xs[m-1], xs[m-1])
        else:
            f, g = make_base(xs[i-1], xs[i], xs[i+1])

        bases0.append(f)
        bases1.append(g)

    return bases0, bases1     

def PiecewiseHermite(xs, ys, dys):
    assert len(xs) == len(ys) == len(dys)
    m = len(xs)

    bases0, bases1 = hermite_bases(xs)

    def f(x):
        res = 0
        for i in range(m):
            res += ys[i] * bases0[i](x) + dys[i] * bases1[i](x)
        return res
    
    return f






