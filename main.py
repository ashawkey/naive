from naive.ode import *
from naive.constants import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

class plotter:
    def __init__(self):
        self.f = plt.figure()
        self.ax = self.f.add_subplot(111)
    
    def plot(self, xs, ys, label):
        self.ax.plot(xs, ys, label=label)
    
    def show(self):
        plt.legend()
        plt.show()

def problem_6_2():
    end_point = 1

    def f(x, y):
        return 100 * y

    fig = plotter()

    for epsilon in [0, 0.0001, 0.001, 0.01, 0.1]:
        xs, ys = euler_forward(f, 0, end_point, 1+epsilon)
        print(f'epsilon={epsilon}, y({end_point})={ys[-1]}')
        fig.plot(xs, ys, str(epsilon))

    fig.show()

def problem_program_6_1():
    def f(x, y):
        return - 1/(x**2) - y/x - y**2
    
    fig = plotter()
    
    xs, ys0 = euler_forward(f, 1, 2, -1)
    fig.plot(xs, ys0, 'euler')

    xs, ys1 = euler_multistep_ac(f, 1, 2, -1)
    fig.plot(xs, ys1, 'euler_improved')

    # ground truth
    from scipy.integrate import ode
    solver = ode(f)
    solver.set_initial_value(-1, 1)
    ys = [-1]
    x = 1
    for x in xs[1:]:
        solver.integrate(x)
        ys.append(solver.y)
    fig.plot(xs, ys, 'ground truth')

    print("error euler: ", np.linalg.norm(np.array(ys0)-np.array(ys), 1))
    print("error euler_improved: ", np.linalg.norm(np.array(ys1)-np.array(ys), 1))


    fig.show()

def problem_program_6_4(subproblem=1):
    def f(t, ys, sigma=10, rou=28, beta=8/3):
        x, y, z = ys[0], ys[1], ys[2]
        return np.array([sigma*(y-x), rou*x-y-x*z, x*y-beta*z])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # (1)
    if subproblem == 1:
        for init_point in [[1,0,0], [100,0,0], [1,1,1], [100,100,100]]:
            xs, ys = runge_kutta(f, 0, 100, np.array(init_point), dx=0.01)
            ys = np.stack(ys)
            ax.plot(ys[:,0], ys[:,1], ys[:,2], label='[' + ','.join([str(x) for x in init_point]) + ']')

    # (2)
    elif subproblem == 2:
        """
        for params in [[10,28,8/3], [10/3, 28/3, 8/9], [30, 84, 8]]:
            xs, ys = runge_kutta(partial(f, sigma=params[0], rou=params[1], beta=params[2]), 0, 50, np.array([1,0,0]), dx=0.01)
            ys = np.stack(ys)
            ax.plot(ys[:,0], ys[:,1], ys[:,2], label='[' + ','.join([str(x) for x in params]) + ']')
        """
        for params in [[10,28,8/3], [10, 28, 0], [10, 0, 8/3], [0, 28, 8/3], [10, 10, 8/3], [10, 99.96, 8/3]]:
            xs, ys = runge_kutta(partial(f, sigma=params[0], rou=params[1], beta=params[2]), 0, 50, np.array([1,1,1]), dx=0.01)
            ys = np.stack(ys)
            ax.plot(ys[:,0], ys[:,1], ys[:,2], label='[' + ','.join([str(x) for x in params]) + ']')
    
    plt.legend()
    plt.show()





if __name__ == "__main__":
    #problem_6_2()
    problem_program_6_1()
    problem_program_6_4(1)
    problem_program_6_4(2)

    