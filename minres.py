import numpy as np

def minres(A, b, epsilon=1e-10, max_it=1000):   #Minimal residual method
    x = np.zeros_like(b)
    x_new = np.zeros_like(b)

    for iteration in range(max_it):
        res = A @ x - b
        A_r = A @ res
        alpha = A_r @ res / (A_r @ A_r)
        x_new = x - alpha*res

        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
        
        x = x_new.copy()

    return x_new

if __name__ == "__main__":
    A = np.array([[4, -1, 1],
                  [-2, 5, 2],
                  [1, -1, 3]])
    
    b = np.array([12, 17, 9])

    solution = minres(A, b)
    print("solution")
    print(solution)