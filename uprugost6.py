import numpy as np
import matplotlib.pyplot as plt

def solve_lame_2d_mkr_simplified(Nx, Ny, Lx, Ly, lambda_lame, mu_lame, Fx_func, Fy_func, u_bc_func, v_bc_func):
    """
    УПРОЩЕННЫЙ СКЕЛЕТ для решения 2D уравнений Ламе МКР.
    ТРЕБУЕТ ЗНАЧИТЕЛЬНОЙ ДОРАБОТКИ И ОТЛАДКИ.
    Предполагаются ГУ Дирихле на всем контуре.
    """
    hx = Lx / Nx
    hy = Ly / Ny
    if not np.isclose(hx, hy):
        print("Внимание: этот упрощенный код лучше работает с hx=hy=h. Результаты могут быть неточными.")
    h = hx # Предположим hx=hy=h для простоты шаблонов

    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)

    # Количество внутренних узлов по каждой координате
    Nx_int = Nx - 1
    Ny_int = Ny - 1
    
    if Nx_int <= 0 or Ny_int <= 0:
        print("Сетка слишком грубая для внутренних узлов.")
        # Нужно вернуть граничные значения или ошибку
        U_sol = np.zeros((Ny + 1, Nx + 1))
        V_sol = np.zeros((Ny + 1, Nx + 1))
        for i_idx in range(Nx+1):
            for j_idx in range(Ny+1):
                U_sol[j_idx, i_idx] = u_bc_func(x[i_idx],y[j_idx])
                V_sol[j_idx, i_idx] = v_bc_func(x[i_idx],y[j_idx])
        return x, y, U_sol, V_sol


    # Общее количество неизвестных: 2 * Nx_int * Ny_int
    # (u-компоненты и v-компоненты во внутренних узлах)
    num_unknowns = 2 * Nx_int * Ny_int
    A_matrix = np.zeros((num_unknowns, num_unknowns))
    b_vector = np.zeros(num_unknowns)

    # Нумерация неизвестных:
    # Сначала все u_ij для внутренних узлов, потом все v_ij для внутренних узлов
    # u_k_u -> u_internal[j_int, i_int]  (k_u от 0 до Nx_int*Ny_int - 1)
    # v_k_v -> v_internal[j_int, i_int]  (k_v от 0 до Nx_int*Ny_int - 1)
    # Глобальный индекс k:
    # для u: k = j_int * Nx_int + i_int
    # для v: k = (Nx_int * Ny_int) + (j_int * Nx_int + i_int)
    
    offset_v = Nx_int * Ny_int

    def map_uv_to_k(j_int, i_int, is_v_component=False):
        base_k = j_int * Nx_int + i_int
        return base_k + offset_v if is_v_component else base_k

    # Коэффициенты из уравнений Ламе
    c1 = lambda_lame + 2 * mu_lame
    c2 = mu_lame
    c3 = lambda_lame + mu_lame

    h2 = h * h # hx*hy если они разные

    # Заполнение матрицы A и вектора b для внутренних узлов
    for j_phys in range(1, Ny): # Физический индекс узла по y (1 до Ny-1)
        for i_phys in range(1, Nx): # Физический индекс узла по x (1 до Nx-1)
            # Индексы внутреннего узла (0-based)
            j_int = j_phys - 1
            i_int = i_phys - 1

            # Индекс уравнения для u-компоненты в текущем узле (i_phys, j_phys)
            k_u_eq = map_uv_to_k(j_int, i_int, is_v_component=False)
            # Индекс уравнения для v-компоненты в текущем узле (i_phys, j_phys)
            k_v_eq = map_uv_to_k(j_int, i_int, is_v_component=True)

            # Правая часть (объемные силы)
            b_vector[k_u_eq] = -Fx_func(x[i_phys], y[j_phys]) * h2 # Умножаем на h2 для удобства
            b_vector[k_v_eq] = -Fy_func(x[i_phys], y[j_phys]) * h2

            # --- Уравнение для u-компоненты (в узле i_phys, j_phys) ---
            # (L+2M)u_xx + M u_yy + (L+M)v_xy = -Fx
            # Коэффициенты при u_{i,j} и v_{i,j} (центральные)
            A_matrix[k_u_eq, map_uv_to_k(j_int, i_int, False)] = -2 * c1 - 2 * c2 # при u_ij
            # A_matrix[k_u_eq, map_uv_to_k(j_int, i_int, True)] = 0 # при v_ij (нет прямого вхождения v_ij)

            # Соседи для u по x (u_xx)
            if i_phys == 1: b_vector[k_u_eq] -= c1 * u_bc_func(x[0], y[j_phys]) # u_{i-1,j} на границе
            else: A_matrix[k_u_eq, map_uv_to_k(j_int, i_int - 1, False)] = c1
            if i_phys == Nx - 1: b_vector[k_u_eq] -= c1 * u_bc_func(x[Nx], y[j_phys]) # u_{i+1,j} на границе
            else: A_matrix[k_u_eq, map_uv_to_k(j_int, i_int + 1, False)] = c1
            
            # Соседи для u по y (u_yy)
            if j_phys == 1: b_vector[k_u_eq] -= c2 * u_bc_func(x[i_phys], y[0]) # u_{i,j-1} на границе
            else: A_matrix[k_u_eq, map_uv_to_k(j_int - 1, i_int, False)] = c2
            if j_phys == Ny - 1: b_vector[k_u_eq] -= c2 * u_bc_func(x[i_phys], y[Ny]) # u_{i,j+1} на границе
            else: A_matrix[k_u_eq, map_uv_to_k(j_int + 1, i_int, False)] = c2

            # Смешанная производная v_xy для u-уравнения: (c3/4) * (v_{i+1,j+1} - v_{i+1,j-1} - v_{i-1,j+1} + v_{i-1,j-1})
            # Это ОЧЕНЬ УПРОЩЕННО И НЕПОЛНО - нужно аккуратно учесть все граничные условия для v
            # при формировании этих членов. Для простоты пропустим здесь детальную реализацию
            # граничных условий для этих членов.

            # ПРИМЕР для v_{i+1,j+1} (если не на границе)
            if i_phys < Nx - 1 and j_phys < Ny - 1:
                 A_matrix[k_u_eq, map_uv_to_k(j_int + 1, i_int + 1, True)] += c3 / 4.0
            # ... и так далее для остальных 3х членов v_xy, аккуратно обрабатывая границы ...


            # --- Уравнение для v-компоненты (в узле i_phys, j_phys) ---
            # M v_xx + (L+2M)v_yy + (L+M)u_xy = -Fy
            A_matrix[k_v_eq, map_uv_to_k(j_int, i_int, True)] = -2 * c2 - 2 * c1 # при v_ij
            
            # Соседи для v по x (v_xx)
            if i_phys == 1: b_vector[k_v_eq] -= c2 * v_bc_func(x[0], y[j_phys])
            else: A_matrix[k_v_eq, map_uv_to_k(j_int, i_int - 1, True)] = c2
            if i_phys == Nx-1: b_vector[k_v_eq] -= c2 * v_bc_func(x[Nx], y[j_phys])
            else: A_matrix[k_v_eq, map_uv_to_k(j_int, i_int + 1, True)] = c2
            
            # Соседи для v по y (v_yy)
            if j_phys == 1: b_vector[k_v_eq] -= c1 * v_bc_func(x[i_phys], y[0])
            else: A_matrix[k_v_eq, map_uv_to_k(j_int - 1, i_int, True)] = c1
            if j_phys == Ny-1: b_vector[k_v_eq] -= c1 * v_bc_func(x[i_phys], y[Ny])
            else: A_matrix[k_v_eq, map_uv_to_k(j_int + 1, i_int, True)] = c1

            # Смешанная производная u_xy для v-уравнения: (c3/4) * (u_{i+1,j+1} - u_{i+1,j-1} - u_{i-1,j+1} + u_{i-1,j-1})
            # ПРИМЕР для u_{i+1,j+1} (если не на границе)
            if i_phys < Nx - 1 and j_phys < Ny - 1:
                 A_matrix[k_v_eq, map_uv_to_k(j_int + 1, i_int + 1, False)] += c3 / 4.0
            # ... и так далее ...
            
    # Решение СЛАУ
    try:
        uv_internal_1d = np.linalg.solve(A_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("2D Ламе: Singular matrix.")
        return x,y,None, None

    # Формирование полных матриц U_sol, V_sol
    U_sol = np.zeros((Ny + 1, Nx + 1))
    V_sol = np.zeros((Ny + 1, Nx + 1))

    # Граничные значения
    for i_idx in range(Nx + 1):
        for j_idx in range(Ny + 1):
            if i_idx == 0 or i_idx == Nx or j_idx == 0 or j_idx == Ny:
                U_sol[j_idx, i_idx] = u_bc_func(x[i_idx], y[j_idx])
                V_sol[j_idx, i_idx] = v_bc_func(x[i_idx], y[j_idx])
    
    # Внутренние значения
    for j_phys in range(1, Ny):
        for i_phys in range(1, Nx):
            j_int, i_int = j_phys - 1, i_phys - 1
            U_sol[j_phys, i_phys] = uv_internal_1d[map_uv_to_k(j_int, i_int, False)]
            V_sol[j_phys, i_phys] = uv_internal_1d[map_uv_to_k(j_int, i_int, True)]
            
    return x, y, U_sol, V_sol

# --- Пример использования (ТРЕБУЕТ АНАЛИТИЧЕСКОГО РЕШЕНИЯ ДЛЯ ТЕСТА) ---
if __name__ == '__main__':
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 10, 10 # ОЧЕНЬ ГРУБАЯ СЕТКА для примера, иначе матрица будет огромной

    lambda_val = 1.0e9 # Примерные значения для Ламе
    mu_val = 0.5e9

    # Тестовая задача: u_exact(x,y), v_exact(x,y)
    # Например, задача о чистом изгибе или простое поле перемещений
    # Для теста нужно подобрать такие u_exact, v_exact, чтобы:
    # 1. Легко вычислялись Fx, Fy (подстановкой u_exact, v_exact в уравнения Ламе)
    # 2. Легко вычислялись u_bc, v_bc (значения u_exact, v_exact на границе)

    # ПРИМЕР: u = sin(pi*x)cos(pi*y), v = cos(pi*x)sin(pi*y) (просто для формы)
    # ЭТО НЕ ОБЯЗАТЕЛЬНО ФИЗИЧЕСКИ КОРРЕКТНОЕ РЕШЕНИЕ УРАВНЕНИЙ ЛАМЕ БЕЗ СПЕЦИАЛЬНЫХ Fx, Fy
    def u_exact_test(x_v, y_v): return np.sin(np.pi*x_v/Lx)*np.cos(np.pi*y_v/Ly)
    def v_exact_test(x_v, y_v): return np.cos(np.pi*x_v/Lx)*np.sin(np.pi*y_v/Ly)

    # Вычисляем Fx, Fy для этих u_exact, v_exact
    # (L+2M)u_xx + M u_yy + (L+M)v_xy = -Fx
    # M v_xx + (L+2M)v_yy + (L+M)u_xy = -Fy
    # Это требует символьного или численного дифференцирования u_exact, v_exact
    # Для простоты, положим Fx=0, Fy=0 и посмотрим, что получится (решение будет u=0,v=0 если ГУ 0)

    def Fx_zero(x_v, y_v): return 0.0
    def Fy_zero(x_v, y_v): return 0.0
    
    # Граничные условия (пусть на границе все закреплено = 0)
    def u_bc_zero(x_v,y_v): return 0.0
    def v_bc_zero(x_v,y_v): return 0.0

    print("Начинаем решение 2D Ламе (упрощенное)...")
    x_g, y_g, U_num, V_num = solve_lame_2d_mkr_simplified(Nx, Ny, Lx, Ly, 
                                                      lambda_val, mu_val,
                                                      Fx_zero, Fy_zero,
                                                      u_bc_zero, v_bc_zero)

    if U_num is not None and V_num is not None:
        print("Решение получено.")
        X_m, Y_m = np.meshgrid(x_g, y_g, indexing='ij')

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X_m.T, Y_m.T, U_num, cmap='viridis')
        ax1.set_title('Перемещение u(x,y)')
        ax1.set_xlabel('x'); ax1.set_ylabel('y')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X_m.T, Y_m.T, V_num, cmap='magma')
        ax2.set_title('Перемещение v(x,y)')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig('lame_2d_mkr_solution.png')
        plt.show()
        
        # Для проверки сходимости нужен метод manufactored solutions:
        # 1. Выбрать u_exact(x,y), v_exact(x,y)
        # 2. Подставить их в уравнения Ламе, чтобы найти Fx(x,y), Fy(x,y)
        # 3. Использовать эти Fx, Fy и значения u_exact, v_exact на границе как ГУ.
        # 4. Сравнивать численное решение с u_exact, v_exact.
    else:
        print("Решение 2D Ламе не получено.")