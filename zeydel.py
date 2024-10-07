import numpy as np

def zeydel(A, b, x0=None, acc=1e-8, max_iterations=100):
    A = np.array(A)
    b = np.array(b)

    n = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)
    print(x)

    for j in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            print(A[i, 1:i:1])
            sum_j = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i:], x[i:]) - A[i, i] * x[i]
            x_new[i] += (b[i] - sum_j) / A[i, i]

            if np.linalg.norm(x_new - x, ord=np.inf) < acc:
                print(f"Iteration number:{j+1}. solved")
                print(x)
                return x_new
            
        x = x_new
        print(f"Iter num:{j+1}")
        print(x)

    print("Max iter")
    return x

if __name__ == "__main__":
    A = np.array([[4, -1, 1],
                  [-2, 5, 2],
                  [1, -1, 3]])
    
    b = np.array([12, 17, 9])

    solution = zeydel(A, b)
    print("solution")
    print(solution)
