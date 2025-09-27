import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def x_poly(x_val, coeffs):
    sum_val = 0
    for i in range(len(coeffs)):
        sum_val += coeffs[i] * x_val**i
    return sum_val

def func(x):
   return np.exp(x)*np.sin(3*x)*np.sin(4*x)

def polynomial_interp(x, f, point):

    n = f.shape[0]
    X = np.array([[xi**i for i in range(n)] for xi in x], dtype=np.double)
    
    a = np.linalg.solve(X, f)

    return x_poly(point, a)

def lagrange_interp(x, f, point):
    result = 0
    n = x.shape[0]

    for i in range(n):
        L_i = 1
        for j in range(n):
         if(i != j):
             L_i *= (point - x[j])/(x[i] - x[j])
        result += f[i]*L_i
       
    return result

x = np.array([0, 1, 2.127615, 3.141592, 4.84255, 6.283185, 7.91497768])
f = np.array([0, 0, 0.64095, 0, -1.564097, 0, 2.064909])
grid = np.arange(np.min(x) - 1, np.max(x) + 2, 0.2)

num = 250

start_time = time.time()
for i in range(num):
    y = polynomial_interp(x, f, grid)
end_time = time.time()
print("Polynom interpolation average execution time: ", (end_time - start_time)/num)

start_time = time.time()
for i in range(num):
    y = lagrange_interp(x, f, grid)
end_time = time.time()
print("Lagrange interpolation average execution time:", (end_time - start_time)/num)

from tqdm import tqdm
from scipy.optimize import curve_fit

def linear_regr(x, a, b):
    return (a*x + b)

def integrate_trapezoid(function, a, b, step=1e-3):
    x = np.arange(a, b, step)
    x = np.append(x, b)
    
    y = function(x)

    sum_trap = np.sum((y[:-1] + y[1:]))* step / 2
    
    return sum_trap

def parab(x):
    return 3 * x**2

integral_value = 2

steps = np.array([1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])
loss = []

for i in steps:
    loss.append(abs(integrate_trapezoid(parab, 0, 1, step=i) - integral_value))

log_loss = np.log(loss)
log_steps = np.log(steps)

popt, pcov = curve_fit(linear_regr, log_steps, log_loss)
grid = np.linspace(-20, 0, 10)

plt.scatter(log_steps, log_loss, color="k")
plt.plot(grid, linear_regr(grid, popt[0], popt[1]), color = "crimson")
plt.xlabel("ln(h)")
plt.ylabel("ln(loss)")
plt.title("Checking the order of convergence")
plt.grid()
plt.savefig("3.png")

print("coeff k: ", popt[0])
    