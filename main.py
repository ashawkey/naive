from naive.optimize import *
import numpy as np
import time
import inspect

class timer:
    def __enter__(self):
        self.time = time.clock()
        return self
    def __exit__(self, type, value, traceback):
        print(f"[TIME] {time.clock() - self.time:.4f}s")

def program_problem_3_2(n=2, subproblem=1):
    if subproblem == 1:
        def f(x):
            '''
            Waston Function. 2 <= n <= 31.
            '''
            res = 0
            for i in range(1, 30):
                r = -1
                for j in range(2, n+1):
                    r += (j - 1) * x[j-1] * (i/29)**(j-2)
                tmp = 0
                for j in range(1, n+1):
                    tmp += x[j-1] * (i/29)**(j-1)
                r -= tmp**2
                res += r**2
            res += x[0]**2
            res += (x[1] - x[0]**2 - 1)**2
            return res
        x0 = np.zeros(n)
        
    elif subproblem == 2:
        h = 1/(n+1)
        def t(i):
            return i * h
        def f(x):
            '''
            Discrete Boundary Value Function
            '''
            res = 0
            for i in range(n):
                if i == 0:
                    r = 2 * x[i] - x[i+1] + h**2 * (x[i] + t(i+1) + 1)**3 / 2
                elif i == n-1:
                    r = 2 * x[i] - x[i-1] + h**2 * (x[i] + t(i+1) + 1)**3 / 2
                else:
                    r = 2 * x[i] - x[i-1] - x[i+1] + h**2 * (x[i] + t(i+1) + 1)**3 / 2
                res += r**2
            return res
        x0 = [t(i+1)*(t(i+1)-1) for i in range(n)]
    
    elif subproblem == 0:
        def f(x):
            '''
            simple test
            '''
            G = np.array([[21,4],[4,15]])
            b = np.array([2,3])
            c = 10
            return x @ G @ x / 2 + b @ x + c
        x0 = [-30, 100]

    for method in [SteepestDescent, 
                   Newton, 
                   partial(QuasiNewton, method='SR1'),
                   partial(QuasiNewton, method='DFP'),
                   partial(QuasiNewton, method='BFGS'),
                   partial(BB, method='BB1'),
                   partial(BB, method='BB2'),
                   ]:
        
        print(f"[METHOD] {method.__name__ if hasattr(method, '__name__') else method.func.__name__}, {inspect.signature(method)}")
        with timer():
            x = method(f, x0)
        print(f"[X] x = {x}")
        print(f"[Y] f(x) = {f(x)}")
        print("")


if __name__ == '__main__':
    #program_problem_3_2(10, 1)
    program_problem_3_2(16, 2)