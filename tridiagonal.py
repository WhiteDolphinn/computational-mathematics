import numpy as np

####        Ax = d
def tridiagonal(sub_diag, diag, sup_diag, d):
    n = len(diag)

    sup_diag_new = [0] * (n-1)
    d_new = [0] * n

    ########first element
    sup_diag_new[0] = sup_diag[0] / diag[0]
    d_new[0] = d[0] / diag[0]


    for i in range(1, n-1):
        sup_diag_new[i] = sup_diag[i] / (diag[i] - sub_diag[i-1] * sup_diag_new[i-1])
        d_new[i] = (d[i] - sub_diag[i-1] * d_new[i-1]) / (diag[i] - sub_diag[i-1] * sup_diag_new[i-1])

    ########last element
    d_new[n-1] = (d[n-1] - sub_diag[n-2] * d_new[n-2]) / (diag[n-1] - sub_diag[n-2] * sup_diag_new[n-2])

    x = [0] * n
    x[n - 1] = d_new[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_new[i] - sup_diag_new[i] * x[i + 1]

    return x
        

if __name__ == "__main__":
    a = [2, 3]
    b = [4, 5, 6]
    c = [1, 2]
    d = [7, 8, 9]

    solution = tridiagonal(a, b, c, d)
    print("Solution:", solution)