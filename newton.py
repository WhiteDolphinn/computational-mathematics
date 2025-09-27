import numpy as np
import unittest
# from sympy import diff
from scipy.misc import derivative

def my_derivative(function, x, dx = 1e-5):
    print("f'(", x, ") =", (function(x+dx) - function(x))/dx)
    return (function(x+dx) - function(x))/dx

def newton(function, x_0 = 1.8, epsilon = 1e-3, iterations = 1000):

    x_old = None
    x = x_0
    x_new = 0
    multiplicity = 1

    for i in range(iterations):
        x_new = x - function(x)/der_polynom(x)

        if(abs(x_new - x) < epsilon):
            break

        if x_old != None:
            multiplicity = 1 / (1 - (x_new - x)/(x - x_old))

        x_old = x
        x = x_new

    return x, multiplicity



   


def polynom(x, a = 1, b = 0, c = 0, d = 0):
    # print("f(", x, ") =", a*x**3 + b*x**2 + c*x + d)
    return a*x**3 + b*x**2 + c*x + d

def der_polynom(x, a = 1, b = 0, c = 0, d = 0):
    return 3*a*x**2 + 2*b*x + c

if __name__== "__main__":
    solution, multiplicity = newton(polynom)
    print("solution: ", solution)
    print("multiplicity: ", multiplicity)

# class TestNewton(unittest.TestCase):
# def test_0(self):
#     def f(x: float) -> float:
#         return x**2 - 20 * sin(x)


#     def f_prime(x: float) -> float:
#         return 2 * x - 20 * cos(x)


#     x0, x_star = 2, 2.7529466338187049383

#     self.assertAlmostEqual(newton(f, f_prime, x0), x_star)




 # x1 = None
    # x = x_0
    # multiplicity = 1

    # for i in range(iterations):
    #     print("f(", x, ") =", function(x))
    #     x_new = x - 24*function(x)/derivative(function, x)

    #     if(abs(x_new - x) < epsilon):
    #         return x_new

    #     if x1 != None:
    #         multiplicity = 1 / (1 - (x_new - x)/(x - x1))
        
    #     x = x_new

    # return x, multiplicity




    # x1 = None
    # x = x_0
    # x_new = 0
    # iter_count = 0
    # multiplicity = 1

    # for i in range(iter_count_max):
    #     x_new = x - function(x)/df(x)

    #     if(abs(x_new - x) < epsilon):
    #         break

    #     if x1 != None:
    #         multiplicity = 1 / (1 - (x_new - x)/(x - x1))

    #     x1 = x
    #     x = x_new
    #     iter_count += 1
    # # x = x_0
    # # multiplicity = 1

    # # for i in range(iterations):
    # #     x_new = x - function(x)/my_derivative(function, x)

    # #     if(abs(x_new - x) < epsilon):
    # #         while abs(function(x_new)) < epsilon:
    # #             multiplicity += 1
    # #             x_new = x - function(x)/my_derivative(function, x)
    # #         print("Кратность корня:", multiplicity)
    # #         return x_new
        
    # #     x = x_new

    # # return x