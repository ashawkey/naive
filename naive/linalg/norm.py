import numpy as np

def norm_1(x):
    return np.sum(np.abs(x))

def norm_2(x):
    return np.sqrt(np.sum(x**2))

def norm_inf(x):
    return np.max(np.abs(x))

def norm_0(x):
    return np.count_nonzero(x)

