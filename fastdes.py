import numpy as np

def fastest_descent(A, b, epsilon=1e-10, max_iterations=1000):
    x = np.zeros_like(b)
    x_new = np.zeros_like(b)

    for i in range(max_iterations):
        res = A @ x - b
        A_r = A @ res
        alpha = res @ res / (A_r @ A_r)
        x_new = x - alpha*res

        if np.linalg.norm(x_new - x) < epsilon:
            print("iteration:", i)
            return x_new
        
        x = x_new.copy()

    print("max_iteration!")
    return x_new

if __name__ == "__main__":
    A = np.array([[4, -1, 1],
                  [-2, 5, 2],
                  [1, -1, 3]])
    
    b = np.array([12, 17, 9])

    solution = fastest_descent(A, b)
    print("solution")
    print(solution)