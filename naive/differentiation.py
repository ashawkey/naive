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