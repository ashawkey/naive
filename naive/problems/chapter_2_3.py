import numpy as np
import scipy.integrate as integrate
from functools import partial
import matplotlib.pyplot as plt

from naive.interpolation import *
from naive.differentiation import *
from naive.integral import *

def problem_2_4():
    xs = np.linspace(0, np.pi/2, 5)
    ys = [np.sin(x) for x in xs]
    # maxR
    def R(x, n=5):
        xs = np.linspace(0, np.pi/2, n)
        res = 1
        for i in range(n):
            res *= (x - xs[i]) / (i + 1)
        return res

    def max_func(f, dx=1e-2):
        return np.max([f(x) for x in np.arange(0, np.pi/2, 1e-1)])

    print("maxR: ", max_func(R))
    
    # test
    maxr = 0
    P = Lagrange(xs, ys)
    for i in range(10):
        x = np.random.rand(1) * np.pi / 2
        maxr = max(maxr, np.sin(x) - P(x))
    print("100 test maxr: ", maxr)
    
    # n for 1e-10
    for n in range(5, 100):
        error = max_func(partial(R, n=n))
        if error < 1e-10:
            print("minimum n: ", n)
            break

def problem_2_16():
    def inner_product(f, g):
        return integrate.quad(lambda x: f(x)*g(x), 0, 1)[0]
    def make_A(k):
        def make_f(n):
            return lambda x: x**n
        fs = [make_f(n) for n in range(k+1)]
        A = np.zeros((k+1, k+1))
        for i in range(k+1):
            A[i, i] = inner_product(fs[i], fs[i])
            for j in range(i):
                A[i, j] = A[j, i] = inner_product(fs[i], fs[j])
        return A
    def kapa(A):
        return (np.linalg.norm(A, ord=2) * np.linalg.norm(np.linalg.inv(A), ord=2))
    
    for k in [5,6,7]:
        print(k, kapa(make_A(k)))

def problem_2():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def plot(f, label):
        xs = np.arange(-5, 5, 1e-2)
        ys = [f(x) for x in xs]
        ax.plot(xs, ys, label=label)
    # Runge
    def Runge(x):
        return 1 / (1 + x**2)
    plot(Runge, 'Runge')
    # Lagrange
    xs = [5*np.cos(np.pi*(2*i+1)/42) for i in range(21)]
    ys = [Runge(x) for x in xs]
    lagrange = Lagrange(xs, ys)
    plot(lagrange, 'Lagrange')
    # Newton
    xs = [-5+i for i in range(11)]
    ys = [Runge(x) for x in xs]
    newton = Newton(xs, ys)
    plot(newton, 'Newton')
    # PiecewiseLinear
    pwl = PiecewiseLinear(xs, ys)
    plot(pwl, 'PiecewiseLinear')
    # PiecewiseHermite
    dys = [diff_central(Runge, x) for x in xs]
    pwh = PiecewiseHermite(xs, ys, dys)
    plot(pwh, 'PiecewiseHermite')

    plt.legend(loc='upper left')
    plt.show()


def problem_3_4():

    def f(x):
        return 4 / (1 + x**2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #dxs = np.linspace(1e-7, 1e-2, 20)[::-1]
    dxs = [0.1,0.01,0.001,0.0001,0.00001]

    ax.set_xlim(dxs[0], dxs[-1])

    for method in ['mid', 'trapezoid', 'simpson']:
        errors = []
        for dx in dxs:
            res = integral(f, 0, 1, method, dx)
            errors.append(np.abs(np.pi - res))
            print(method, dx, res, np.abs(np.pi - res))
        ax.plot(dxs, errors, '-o', label=method)
    
    #plt.xticks(list(range(len(dxs))), dxs)
    plt.legend(loc='upper left')
    plt.show()

def problem_3_10():
    def f(x, y):
        return np.e**(-x*y)
    print(double_integral(f, 0, 1, lambda x:0, lambda x:1, method='trapezoid', dx=0.01),
          integrate.dblquad(f, 0, 1, lambda x:0, lambda x:1))
    print(double_integral(f, 0, 1, lambda x:0, lambda x:np.sqrt(1-x**2), method='trapezoid', dx=0.01),
          integrate.dblquad(f, 0, 1, lambda x:0, lambda x:np.sqrt(1-x**2)))

if __name__ == "__main__":
    #problem_2_4()
    #problem_2_16()
    #problem_2()
    problem_3_4()
    #problem_3_10()
    