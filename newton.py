import numpy as np

def derivative(function, x, dx = 1e-5):
    print("f'(", x, ") =", (function(x+dx) - function(x))/dx)
    return (function(x+dx) - function(x))/dx

def newton(function, x_0 = -0.02, epsilon = 1e-3, iterations = 1000):
    x = x_0

    for i in range(iterations):
        x_new = x - function(x)/derivative(function, x)

        if(abs(x_new - x) < epsilon):
            return x_new
        
        x = x_new

    return x


def polynom(x, a = 1, b = 1, c = 0, d = -0.1):
    print("f(", x, ") =", a*x**3 + b*x**2 + c*x + d)
    return a*x**3 + b*x**2 + c*x + d

if __name__== "__main__":
    solution = newton(polynom)
    print("solution: ", solution)