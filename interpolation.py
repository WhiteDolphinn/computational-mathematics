import numpy as np
import time
import matplotlib.pyplot as plt 

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

x = np.linspace(0, 3, 11)
f = func(x)

grid = np.arange(np.min(x) - 1, np.max(x) + 1, 0.0001)
y = polynomial_interp(x, f, grid)
ground_truth = func(grid)

plt.plot(grid, y, label = "Polynom interpolation", color = "green")
plt.scatter(x, f,  label = "initial points", color = "red")
plt.plot(grid, ground_truth, label = "initial function", color = "crimson")
plt.grid()
plt.title("Polynom interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-20, 20)
# plt.ylim((min(y) - 1, max(y) + 1))
plt.legend()
plt.savefig("1.png")
plt.close()




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

x = np.linspace(0, 3, 8)
f = func(x)

grid = np.arange(np.min(x) - 1, np.max(x) + 1, 0.0001)
y = lagrange_interp(x, f, grid)
ground_truth = func(grid)

plt.plot(grid, y, label = "Lagrange interpolation", color="green")
plt.scatter(x, f,  label = "initial points", color = "red")
plt.plot(grid, ground_truth, label = "initial function", color="crimson")
plt.plot()
plt.grid()
plt.ylim((-2, 2))
plt.title("Lagrange polynom interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("2.png")
# plt.close()

    