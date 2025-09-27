import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.optimize import root_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
c = lambda: randint(0, 10)*h**2 / 20 + 1 * h ** 2

def analytical(x):
    return (np.exp((6 + np.sqrt(13))*x) - np.exp((6 - np.sqrt(13))*x)) / (2*np.sqrt(13))

def shooting_method(order, h, y_right):
    
    def problem(x, y, dydx):
        return 12*dydx - 23*y
    
    def solve_ivp(s_guess):
        x_vals = np.arange(0, 1 + h, h)
        y = 0.0       # y(0) = 0
        dydx = s_guess  # y'(0) = s_guess
        
        y_vals = [y]
        
        for i in range(1, len(x_vals)):
            x_prev = x_vals[i-1]
            
            if order == 1:
                y_new = y + h * dydx
                dydx_new = dydx + h * problem(x_prev, y, dydx)
                
            elif order == 2:
                k1_y = dydx
                k1_dydx = problem(x_prev, y, dydx)
                
                k2_y = dydx + h * k1_dydx
                k2_dydx = problem(x_prev + h, y + h * k1_y, dydx + h * k1_dydx)
                
                y_new = y + (h / 2) * (k1_y + k2_y)
                dydx_new = dydx + (h / 2) * (k1_dydx + k2_dydx)
                
            elif order == 4:
                k1_y = dydx
                k1_dydx = problem(x_prev, y, dydx)
                
                k2_y = dydx + (h / 2) * k1_dydx
                k2_dydx = problem(x_prev + h/2, y + (h/2)*k1_y, dydx + (h/2)*k1_dydx)
                
                k3_y = dydx + (h / 2) * k2_dydx
                k3_dydx = problem(x_prev + h/2, y + (h/2)*k2_y, dydx + (h/2)*k2_dydx)
                
                k4_y = dydx + h * k3_dydx
                k4_dydx = problem(x_prev + h, y + h*k3_y, dydx + h*k3_dydx)
                
                y_new = y + (h / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
                dydx_new = dydx + (h / 6) * (k1_dydx + 2*k2_dydx + 2*k3_dydx + k4_dydx)
            
            y, dydx = y_new, dydx_new
            y_vals.append(y)
        
        return y_vals[-1] - y_right
    
    # Автоматический подбор интервала
    a, b = -1.0, 1.0  # Начальные границы
    fa, fb = solve_ivp(a), solve_ivp(b)
    
    # Расширяем интервал, пока не получим разные знаки
    for _ in range(100):
        if fa * fb < 0:
            break
        if abs(fa) < abs(fb):
            a *= 2.0
            fa = solve_ivp(a)
        else:
            b *= 2.0
            fb = solve_ivp(b)
    else:
        raise ValueError("Не удалось найти интервал с разными знаками")

    # Находим правильное начальное значение производной
    sol = root_scalar(solve_ivp, bracket=[a, b], method='brentq')
    s_optimal = sol.root
    
    # Решаем с оптимальным параметром
    x_vals = np.arange(0, 1 + h, h)
    y = 0.0
    dydx = s_optimal
    
    y_vals = [y]
    
    for i in range(1, len(x_vals)):
        x_prev = x_vals[i-1]
        
        if order == 1:
            y_new = y + h * dydx
            dydx_new = dydx + h * problem(x_prev, y, dydx)
            
        elif order == 2:
            k1_y = dydx
            k1_dydx = problem(x_prev, y, dydx)
            
            k2_y = dydx + h * k1_dydx
            k2_dydx = problem(x_prev + h, y + h * k1_y, dydx + h * k1_dydx)
            
            y_new = y + (h / 2) * (k1_y + k2_y)
            dydx_new = dydx + (h / 2) * (k1_dydx + k2_dydx)
            
        elif order == 4:
            k1_y = dydx
            k1_dydx = problem(x_prev, y, dydx)
            
            k2_y = dydx + (h / 2) * k1_dydx
            k2_dydx = problem(x_prev + h/2, y + (h/2)*k1_y, dydx + (h/2)*k1_dydx)
            
            k3_y = dydx + (h / 2) * k2_dydx
            k3_dydx = problem(x_prev + h/2, y + (h/2)*k2_y, dydx + (h/2)*k2_dydx)
            
            k4_y = dydx + h * k3_dydx
            k4_dydx = problem(x_prev + h, y + h*k3_y, dydx + h*k3_dydx)
            
            y_new = y + (h / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            dydx_new = dydx + (h / 6) * (k1_dydx + 2*k2_dydx + 2*k3_dydx + k4_dydx)
        
        y, dydx = y_new, dydx_new
        y_vals.append(y)
    
    return x_vals, np.array(y_vals)

def finite_difference_method(a, b, ya, yb, h):
    """
    Решает краевую задачу y'' = 12*y' - 23*y с условиями:
    y(a) = ya, y(b) = yb
    методом конечных разностей.
    """
    N = int((b - a)/h)
    x = np.linspace(a, b, N+1)
    
    # Коэффициенты для разностных схем:
    # y'' ≈ (y[i+1] - 2y[i] + y[i-1])/h² (центральная разность 2-го порядка)
    # y' ≈ (y[i+1] - y[i-1])/(2h) (центральная разность 2-го порядка)
    
    main_diag = (-2/h**2 + 23) * np.ones(N-1)
    lower_diag = (1/h**2 + 12/(2*h)) * np.ones(N-2)
    upper_diag = (1/h**2 - 12/(2*h)) * np.ones(N-2)
    
    
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    # print(A)
    
    F = np.zeros(N-1)
    # print(f)

    
    F[0] -= ya * (1/h**2 + 12/(2*h))
    F[-1] -= yb * (1/h**2 - 12/(2*h))
    
    y_interior = spsolve(A, F)
    y = np.concatenate([[ya], y_interior, [yb]])
    
    return x, y

hs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
errors_rk1, errors_rk2, errors_rk4, errors_fd = [], [], [], []

for h in hs:
    y_r = (np.exp(6 + np.sqrt(13)) - np.exp(6 - np.sqrt(13))) / (2*np.sqrt(13))
    x_rk1, y_rk1 = shooting_method(1, h, y_r)
    x_rk2, y_rk2 = shooting_method(2, h, y_r)
    x_rk4, y_rk4 = shooting_method(4, h, y_r)
    x_fd, y_fd = finite_difference_method(0, 1, 0, y_r, h)

    y_exact = analytical(x_rk1)
    errors_rk1.append(np.max(np.abs(y_exact - y_rk1)))
    errors_rk2.append(np.max(np.abs(y_exact - y_rk2)))
    errors_rk4.append(np.max(np.abs(y_exact - y_rk4)))
    errors_fd.append(np.max(np.abs(y_exact - y_fd)))

plt.figure(figsize=(8, 6))
plt.loglog(hs, errors_rk1, 'o-', label='shooting1', color = 'r')

plt.loglog(hs, errors_rk2, 's-', label='shooting2', color = 'g')

plt.loglog(hs, errors_rk4, '^-', label='shooting4', color = 'b')

plt.loglog(hs, errors_fd, 'd-', label='finite differences', color = 'olive')
plt.xlabel("log(h)")
plt.ylabel("log(error)")
plt.legend()
plt.grid(True)
plt.savefig("shooting.png")
# plt.show()

for h in hs:
    x_rk1, y_rk1 = shooting_method(1, h, y_r)
    x_rk2, y_rk2 = shooting_method(2, h, y_r)
    x_rk4, y_rk4 = shooting_method(4, h, y_r)
    x_fd, y_fd = finite_difference_method(0, 1, 0, y_r, h)


    plt.figure(figsize=(8, 6))
    x2 = np.arange(0, 1, 0.0001)
    plt.errorbar(x2, analytical(x2), 0, color='crimson', label='analytical')
    plt.errorbar(x_rk1, y_rk1, 0, color='purple',  label='shooting1')
    plt.errorbar(x_rk2, y_rk2, 0, color='olive',  label='shooting2')
    plt.errorbar(x_rk4, y_rk4, 0, color='gold',  label='shooting4')
    plt.errorbar(x_fd, y_fd, 0, color='blue', label='finite difference')
    filename = f'shooting2_{h}.png'
    plt.savefig(filename)
    # plt.savefig("shooting2.png")
    # plt.show()