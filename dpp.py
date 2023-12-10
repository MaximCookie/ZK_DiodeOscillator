import numpy as np
import matplotlib.pyplot as plt

# global constants
delta = 0.03
epsilon = 10**-10

def rungeKutta4(x0, y0, step, n, func):
    res = []
    y = y0
    x = x0
    for _ in range(1, n + 1):
        k1 = step * func(x, y)
        k2 = step * func(x + 0.5 * step, y + 0.5 * k1)
        k3 = step * func(x + 0.5 * step, y + 0.5 * k2)
        k4 = step * func(x + step, y + k3)

        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        res.append(y)
        x = x + step

    return np.array(res)

# initial system
def dxdt(t, x):
    res = (x*np.sin(t) - x**2 + epsilon) / delta
    return np.array(res)

# best argument system
def dSdl(l, S):
    x, t = S
    res = [(x*np.sin(t) - x**2 + epsilon) / (np.sqrt(delta**2 + (x*np.sin(t) - x**2 + epsilon)**2)),
           delta / np.sqrt(delta**2 + (x*np.sin(t) - x**2 + epsilon)**2)]
    return np.array(res)

l, r = map(float, input('l, r: ').split())
n = int(input('n: '))
step = (r - l) / n 
args = np.linspace(l, r, n)

a_n = int(input('number of graphs: '))
a_vals = np.linspace(0.1, 0.9, a_n)
for a in a_vals:
    x_0 = 0
    y_0 = a
    S_0 = (y_0, x_0)

    base_sol = rungeKutta4(x_0, y_0, step, n, dxdt)
    param_sol = rungeKutta4(x_0, S_0, step, n, dSdl)
    plt.subplot(121)
    plt.plot(args, base_sol, label=f'a={round(a, 2)}')
    plt.subplot(122)
    plt.plot(param_sol.T[1], param_sol.T[0], label=f'a={round(a, 2)}')
  
plt.subplot(121)
plt.legend(loc='upper right')
plt.title('Начальная система')

plt.subplot(122)
plt.legend(loc='upper right')
plt.title('Продолжение по параметру')

plt.show()