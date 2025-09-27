import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def solve_poisson_2d_mkr(Nx, Ny, Lx, Ly, f_func, g_func_boundary):
    """
    Решает 2D уравнение Пуассона u_xx + u_yy = f(x,y) на прямоугольнике [0,Lx]x[0,Ly]
    с граничными условиями Дирихле u = g(x,y) на границе.
    Использует метод конечных разностей.

    Nx, Ny: количество интервалов по x и y
    Lx, Ly: размеры области
    f_func(x,y): правая часть уравнения
    g_func_boundary(x,y): значения на границе
    """
    hx = Lx / Nx
    hy = Ly / Ny
    
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y) # Для вычисления f и g

    # Количество внутренних узлов
    num_internal_nodes = (Nx - 1) * (Ny - 1)
    if num_internal_nodes <= 0:
        print("Сетка слишком грубая, нет внутренних узлов.")
        # Просто вернем значения на границе
        U_sol = np.zeros((Ny + 1, Nx + 1))
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                if i == 0 or i == Nx or j == 0 or j == Ny:
                    U_sol[j, i] = g_func_boundary(x[i], y[j])
        return x, y, U_sol


    A_matrix = np.zeros((num_internal_nodes, num_internal_nodes))
    b_vector = np.zeros(num_internal_nodes)

    # Функция для отображения 2D индекса (j_int, i_int) в 1D индекс k
    # Внутренние узлы: i_int от 0 до Nx-2, j_int от 0 до Ny-2
    # Соответствуют физическим узлам i от 1 до Nx-1, j от 1 до Ny-1
    def map_to_k(j_int, i_int):
        return j_int * (Nx - 1) + i_int

    # Заполнение матрицы A и вектора b
    for j_phys in range(1, Ny): # Физический индекс по y (1 до Ny-1)
        for i_phys in range(1, Nx): # Физический индекс по x (1 до Nx-1)
            # Индексы для внутренних узлов (для матрицы)
            j_int = j_phys - 1
            i_int = i_phys - 1
            k = map_to_k(j_int, i_int) # Текущий 1D индекс уравнения

            # Коэффициенты для пятиточечного шаблона (hx=hy=h для простоты, здесь разные)
            # (u_i+1,j - 2u_ij + u_i-1,j)/hx^2 + (u_i,j+1 - 2u_ij + u_i,j-1)/hy^2 = f_ij
            # Перегруппируем:
            # u_i-1,j / hx^2 + u_i+1,j / hx^2 + 
            # u_i,j-1 / hy^2 + u_i,j+1 / hy^2 - 
            # u_ij * (2/hx^2 + 2/hy^2) = f_ij

            A_matrix[k, k] = -2.0 / (hx**2) - 2.0 / (hy**2)
            b_vector[k] = f_func(x[i_phys], y[j_phys])

            # Соседи
            # u_{i-1, j}
            if i_phys == 1: # Граница слева u(0, y_j)
                b_vector[k] -= g_func_boundary(x[0], y[j_phys]) / (hx**2)
            else:
                A_matrix[k, map_to_k(j_int, i_int - 1)] = 1.0 / (hx**2)
            
            # u_{i+1, j}
            if i_phys == Nx - 1: # Граница справа u(Lx, y_j)
                b_vector[k] -= g_func_boundary(x[Nx], y[j_phys]) / (hx**2)
            else:
                A_matrix[k, map_to_k(j_int, i_int + 1)] = 1.0 / (hx**2)

            # u_{i, j-1}
            if j_phys == 1: # Граница снизу u(x_i, 0)
                b_vector[k] -= g_func_boundary(x[i_phys], y[0]) / (hy**2)
            else:
                A_matrix[k, map_to_k(j_int - 1, i_int)] = 1.0 / (hy**2)

            # u_{i, j+1}
            if j_phys == Ny - 1: # Граница сверху u(x_i, Ly)
                b_vector[k] -= g_func_boundary(x[i_phys], y[Ny]) / (hy**2)
            else:
                A_matrix[k, map_to_k(j_int + 1, i_int)] = 1.0 / (hy**2)
                
    # Решение СЛАУ
    try:
        u_internal_1d = np.linalg.solve(A_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("2D Poisson: Singular matrix.")
        return x,y,None


    # Формирование полной матрицы решения U_sol (Ny+1, Nx+1)
    U_sol = np.zeros((Ny + 1, Nx + 1))

    # Заполнение граничных значений
    for i in range(Nx + 1):
        U_sol[0, i] = g_func_boundary(x[i], y[0])
        U_sol[Ny, i] = g_func_boundary(x[i], y[Ny])
    for j in range(Ny + 1):
        U_sol[j, 0] = g_func_boundary(x[0], y[j])
        U_sol[j, Nx] = g_func_boundary(x[Nx], y[j])

    # Заполнение внутренних значений
    for j_phys in range(1, Ny):
        for i_phys in range(1, Nx):
            j_int = j_phys - 1
            i_int = i_phys - 1
            k = map_to_k(j_int, i_int)
            U_sol[j_phys, i_phys] = u_internal_1d[k]
            
    return x, y, U_sol

# --- Пример использования для 2D Пуассона ---
Lx_2d, Ly_2d = 1.0, 1.0
Nx_2d, Ny_2d = 20, 20 # Количество интервалов

# Тестовая задача: u_exact(x,y) = sin(pi*x/Lx) * sin(pi*y/Ly)
# Тогда u_xx + u_yy = -(pi/Lx)^2 u - (pi/Ly)^2 u = -((pi/Lx)^2 + (pi/Ly)^2) * sin(pi*x/Lx)sin(pi*y/Ly)
def f_exact_2d(x_val, y_val):
    term_x = (np.pi / Lx_2d)**2
    term_y = (np.pi / Ly_2d)**2
    return -(term_x + term_y) * np.sin(np.pi * x_val / Lx_2d) * np.sin(np.pi * y_val / Ly_2d)

def g_boundary_exact_2d(x_val, y_val): # Значения на границе
    # На границах x=0, x=Lx, y=0, y=Ly это будет 0
    if np.isclose(x_val, 0.0) or np.isclose(x_val, Lx_2d) or \
       np.isclose(y_val, 0.0) or np.isclose(y_val, Ly_2d):
        return 0.0 
    # Это условие нужно для точного g. В нашем случае, если u_exact - синус, то на границе 0.
    return np.sin(np.pi * x_val / Lx_2d) * np.sin(np.pi * y_val / Ly_2d) # Для внутренних (не используется)

def u_analytical_2d(x_val, y_val):
     return np.sin(np.pi * x_val / Lx_2d) * np.sin(np.pi * y_val / Ly_2d)

x_grid_2d, y_grid_2d, U_numerical_2d = solve_poisson_2d_mkr(Nx_2d, Ny_2d, Lx_2d, Ly_2d, 
                                                             f_exact_2d, g_boundary_exact_2d)

if U_numerical_2d is not None:
    X_grid, Y_grid = np.meshgrid(x_grid_2d, y_grid_2d, indexing='ij') # Важно для meshgrid
    
    U_analytical_on_grid = u_analytical_2d(X_grid, Y_grid)

    # 3D Поверхностный график численного решения
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X_grid.T, Y_grid.T, U_numerical_2d, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y) - численное')
    ax1.set_title(f'2D Пуассон: Численное решение (МКР, N={Nx_2d}x{Ny_2d})')
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    plt.savefig('poisson_2d_numerical.png')
    plt.show()

    # 3D Поверхностный график аналитического решения
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(X_grid.T, Y_grid.T, U_analytical_on_grid, cmap=cm.magma, linewidth=0, antialiased=False)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y) - аналитическое')
    ax2.set_title('2D Пуассон: Аналитическое решение')
    fig2.colorbar(surf2, shrink=0.5, aspect=5)
    plt.savefig('poisson_2d_analytical.png')
    plt.show()

    # Контурный график ошибки
    error_2d = np.abs(U_numerical_2d - U_analytical_on_grid)
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    contour = ax3.contourf(X_grid.T, Y_grid.T, error_2d, cmap="coolwarm", levels=20)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'2D Пуассон: Абсолютная ошибка (МКР, N={Nx_2d}x{Ny_2d})')
    fig3.colorbar(contour)
    ax3.set_aspect('equal', adjustable='box')
    plt.savefig('poisson_2d_error_contour.png')
    plt.show()
    
    print(f"Максимальная абсолютная ошибка для 2D Пуассона: {np.max(error_2d):.2e}")

else:
    print("Не удалось получить 2D решение.")