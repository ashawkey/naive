import numpy as np
from naive.differentiation import diff
from naive.constants import *

def euler_forward(f, x0, x1, y0, dx=DX):
    """
    solve ode problem in [x0, x1]:
        dy/dx = f(x, y)
        y(x0) = y0
    """
    xs = [x0]
    ys = [y0]
    for i in range(1, int((x1-x0)//dx)+1):
        xi = xs[i-1] + dx
        xs.append(xi)
        yi = ys[i-1] + dx * f(xs[i-1], ys[i-1])
        ys.append(yi)
    
    return xs, ys

def euler_backward(f, x0, x1, y0, dx=DX):
    """
    implicit eluer. 
    """
    pass

def runge_kutta(f, x0, x1, y0, method='runge_kutta_4', dx=DX):
    xs = [x0]
    ys = [y0]
    for i in range(1, int((x1-x0)//dx)+1):
        xi = xs[-1] + dx
        xs.append(xi)
        if method == 'improved_euler':
            k1 = f(xs[i-1], ys[i-1])
            k2 = f(xs[i-1] + dx, ys[i-1] + dx * k1)
            yi = ys[i-1] + dx/2 * (k1 + k2)
        elif method == 'heun_2':
            k1 = f(xs[i-1], ys[i-1])
            k2 = f(xs[i-1] + 2/3 * dx, ys[i-1] + 2/3 * dx * k1)
            yi = ys[i-1] + dx/4 * (k1 + 3*k2)
        elif method == 'runge_kutta_4':
            k1 = f(xs[i-1], ys[i-1])
            k2 = f(xs[i-1] + dx/2, ys[i-1] + dx/2 * k1)
            k3 = f(xs[i-1] + dx/2, ys[i-1] + dx/2 * k2)
            k4 = f(xs[i-1] + dx, ys[i-1] + dx * k3)
            yi = ys[i-1] + dx/6 * (k1 + 2*k2 + 2*k3 + k4)            
        else:
            raise NotImplementedError
        ys.append(yi)
    
    return xs, ys

def euler_multistep_ac(f, x0, x1, y0, dx=DX):
    """
    implicit approximate-calibrate two-step
    (also called improved_euler ?)
    """
    xs = [x0]
    ys = [y0]
    for i in range(1, int((x1-x0)//dx)+1):
        xi = xs[i-1] + dx
        xs.append(xi)
        yi_ = ys[i-1] + dx * f(xs[i-1], ys[i-1])
        yi = ys[i-1] + dx/2 * (f(xs[i-1], ys[i-1]) + f(xi, yi_))
        ys.append(yi)
    
    return xs, ys    


