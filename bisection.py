import numpy as np

def bisection(function, a, b, epsilon=1e-5):
    left_val = function(a)
    right_val = function(b)

    if(left_val * right_val > 0):
        return ValueError("Function must have different signs in both ends. (f(a) = ", left_val, ", f(b) = ", right_val, " ).")
    
    center = (a + b) / 2
    center_val = function(center)

    if(left_val * center_val < 0):
        if((center - a) < epsilon):
            return ((a + center) / 2)
        
        root = bisection(function, a, center, epsilon)

    else:
        if((b - center) < epsilon):
            return ((center + b) / 2)
        
        root = bisection(function, center, b, epsilon)

    return root

def polynom(x, a = 1, b = 1, c = 0, d = -1):
    print("f(", x, ") = ", a*x**3 + b*x**2 + c*x + d)
    return a*x**3 + b*x**2 + c*x + d

if __name__ == "__main__":
    solution = bisection(polynom, 0, 1, 1e-5)
    print("solution", solution)
