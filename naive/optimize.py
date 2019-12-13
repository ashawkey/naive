import numpy as np
from functools import partial
from .differentiation import *
from .linalg.inversion import inverse

def golden(f, l=-1000, r=1000, max_step=MAX_STEP, tau=0.618, epsilon=0.01, verbose=True):
    '''
    1-dim minimum search of f(x) in [l, r]
    '''
    step = 0
    while True:
        step += 1
        if step > max_step:
            if verbose:
                print(f"Exceed max_step {max_step}, force stop")
            return (l + r) / 2
        elif r - l < epsilon:
            return (l + r) / 2
        else:
            ll = l + (1 - tau) * (r - l)
            rr = l + tau * (r - l)
            if f(ll) < f(rr):
                l, r = l, rr
            else:
                l, r = ll, r

def SteepestDescent(f, x0, max_step=MAX_STEP, epsilon=1e-5, search_function=partial(golden, l=0, r=100), verbose=True):
    """
    Minimize Multi-variable function f.
    Must give x0 which reveals the dimension of f.
    """
    step = 0
    x = x0
    while True:
        step += 1
        if step > max_step:
            if verbose:
                print(f"Exceed max_step {max_step}, force stop")
            return x
        else:
            g = gradient(f, x)
            if np.linalg.norm(g, 1) < epsilon:
                if verbose:
                    print(f"Convergent at step {step} with epsilon {epsilon}")
                return x
            d = - g
            alpha = search_function(lambda alpha: f(x + alpha * d))
            x = x + alpha * d


def Newton(f, x0, max_step=MAX_STEP, epsilon=1e-5, verbose=True):
    step = 0
    x = x0
    while True:
        step += 1
        if step > max_step:
            if verbose:
                print(f"Exceed max_step {max_step}, force stop")
            return x
        else:
            g = gradient(f, x)
            if np.linalg.norm(g, 1) < epsilon:
                if verbose:
                    print(f"Convergent at step {step} with epsilon {epsilon}")
                return x
            G = hessian(f, x)
            d = - inverse(G) @ g
            x = x + d

def QuasiNewton(f, x0, H0=None, max_step=MAX_STEP, epsilon=1e-5, verbose=True,
                search_function=partial(golden, l=0, r=100), method='SR1', broyden_phi=0.5):
    dim = len(x0)
    step = 0
    x = x0
    H = H0 if H0 is not None else np.eye(dim)
    g = gradient(f, x)
    while True:
        step += 1
        if step > max_step:
            if verbose:
                print(f"Exceed max_step {max_step}, force stop")
            return x
        else:
            if np.linalg.norm(g, 1) < epsilon:
                if verbose:
                    print(f"Convergent at step {step} with epsilon {epsilon}")
                return x
            d = - H @ g
            alpha = search_function(lambda alpha: f(x + alpha * d))            
            
            s = alpha * d
            x = x + s

            old_g = np.copy(g)
            g = gradient(f, x)
            y = g - old_g

            if method == 'SR1':
                tmp = (s - H @ y)
                dH = (tmp.reshape(-1,1) @ tmp.reshape(1,-1)) / (tmp @ y)
            elif method == 'DFP':
                dH = (s.reshape(-1,1) @ s.reshape(1,-1)) / (s @ y) - (H @ (y.reshape(-1,1) @ y.reshape(1,-1)) @ H) / (y @ H @ y)
            elif method == 'BFGS':
                dH = ((1 + (y @ H @ y) / (y @ s)) * (s.reshape(-1,1) @ s.reshape(1,-1)) / (y @ s)) - ((s.reshape(-1,1) @ y.reshape(1,-1) @ H + H @ y.reshape(-1,1) @ s.reshape(1,-1)) / (y @ s))
            elif method == 'Broyden':
                dH_dfp = (s.reshape(-1,1) @ s.reshape(1,-1)) / (s @ y) - (H @ (y.reshape(-1,1) @ y.reshape(1,-1)) @ H) / (y @ H @ y)
                dH_bfgs = ((1 + (y @ H @ y) / (y @ s)) * (s.reshape(-1,1) @ s.reshape(1,-1)) / (y @ s)) - ((s.reshape(-1,1) @ y.reshape(1,-1) @ H + H @ y.reshape(-1,1) @ s.reshape(1,-1)) / (y @ s))
                dH = (1 - broyden_phi) * dH_dfp + broyden_phi * dH_bfgs
            H = H + dH


def BB(f, x0, max_step=MAX_STEP, epsilon=1e-5, method='BB1', verbose=True, search_function=partial(golden, l=0, r=100)):
    dim = len(x0)
    step = 0
    x = x0
    g = gradient(f, x)
    old_x = None
    old_g = None
    while True:
        step += 1
        if step > max_step:
            if verbose:
                print(f"Exceed max_step {max_step}, force stop")
            return x
        else:
            if np.linalg.norm(g, 1) < epsilon:
                if verbose:
                    print(f"Convergent at step {step} with epsilon {epsilon}")
                return x
            d = - g
            if old_x is None:
                alpha = search_function(lambda alpha: f(x + alpha * d))
            else:
                s = x - old_x
                y = g - old_g
                if method == 'BB1':
                    alpha = (s @ s) / (s @ y)
                elif method == 'BB2':
                    alpha = (s @ y) / (y @ y)
                else:
                    raise NotImplementedError
            old_x = x
            x = x + alpha * d
            old_g = g
            g = gradient(f, x)
            