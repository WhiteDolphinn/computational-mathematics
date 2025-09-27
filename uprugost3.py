import numpy as np
import matplotlib.pyplot as plt

# --- Параметры задачи (те же, что и для МКР) ---
L = 100.0
E = 2.1e11
A = 1e-4
P = 1e4
f_amplitude = 1e3

def f_sin_load(x_coord): # Синусоидальная нагрузка
    return f_amplitude * np.sin(np.pi * x_coord / L)

def analytical_solution_sin_load(x, L, E, A, P, f_amp): # Аналитическое решение
    C1 = (P + f_amp * L / np.pi) / (E * A)
    C2 = 0
    u_particular_due_to_f = (f_amp * L**2) / (E * A * np.pi**2) * np.sin(np.pi * x / L)
    u_general_homogeneous = C1 * x + C2
    return u_particular_due_to_f + u_general_homogeneous

# --- Численное решение (Метод Конечных Элементов) ---
def solve_1d_rod_fem(num_elements, L, E, A, P_force, f_func):
    num_nodes = num_elements + 1
    x_nodes = np.linspace(0, L, num_nodes)

    # Глобальная матрица жесткости и вектор сил
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)

    # Цикл по элементам для сборки K_global и F_global
    for e in range(num_elements):
        node1_idx = e
        node2_idx = e + 1
        
        x1 = x_nodes[node1_idx]
        x2 = x_nodes[node2_idx]
        h_e = x2 - x1 # Длина элемента

        # Локальная матрица жесткости для стержневого элемента
        k_e_local = (E * A / h_e) * np.array([[1, -1],
                                              [-1, 1]])
        
        # Сборка в глобальную матрицу жесткости
        K_global[node1_idx, node1_idx] += k_e_local[0, 0]
        K_global[node1_idx, node2_idx] += k_e_local[0, 1]
        K_global[node2_idx, node1_idx] += k_e_local[1, 0]
        K_global[node2_idx, node2_idx] += k_e_local[1, 1]

        # Локальный вектор сил от распределенной нагрузки f(x)
        # Используем численное интегрирование (правило трапеций для простоты на элементе)
        # f_e_local = [ integral(N1*f dx), integral(N2*f dx) ]
        # N1(x') = 1 - x'/h_e, N2(x') = x'/h_e, где x' - локальная координата от 0 до h_e
        # Если f(x) аппроксимируется линейно на элементе: f(x') = N1*f1 + N2*f2
        # integral(N1*(N1*f1+N2*f2))dx' = h_e/6 * (2*f1 + f2)
        # integral(N2*(N1*f1+N2*f2))dx' = h_e/6 * (f1 + 2*f2)
        f_at_node1 = f_func(x1)
        f_at_node2 = f_func(x2)
        
        f_e_local_contrib = (h_e / 6.0) * np.array([2 * f_at_node1 + f_at_node2,
                                                   f_at_node1 + 2 * f_at_node2])
        
        F_global[node1_idx] += f_e_local_contrib[0]
        F_global[node2_idx] += f_e_local_contrib[1]

    # Применение граничных условий
    # 1. Силовое ГУ (естественное): P на правом конце (узел num_nodes-1)
    F_global[num_nodes - 1] += P_force

    # 2. Кинематическое ГУ (главное): u(0) = 0 (узел 0)
    # Метод исключения:
    # Сохраняем первую строку и столбец для последующего восстановления
    # K_reduced = K_global[1:, 1:]
    # F_reduced = F_global[1:] - K_global[1:, 0] * 0.0 # u_0 = 0
    # U_reduced = np.linalg.solve(K_reduced, F_reduced)
    # U_fem = np.zeros(num_nodes)
    # U_fem[1:] = U_reduced
    
    # Более общий подход (метод больших чисел или модификация матрицы)
    # Для u_0 = 0:
    # Обнуляем 0-ю строку и 0-й столбец K_global, кроме K_global[0,0]=1
    # F_global[0] = 0
    K_modified = np.copy(K_global)
    F_modified = np.copy(F_global)

    K_modified[0, :] = 0.0
    K_modified[:, 0] = 0.0
    K_modified[0, 0] = 1.0
    F_modified[0] = 0.0 # Заданное значение перемещения u_0

    # Решение СЛАУ
    try:
        U_fem = np.linalg.solve(K_modified, F_modified)
    except np.linalg.LinAlgError:
        print(f"FEM: Singular matrix for num_elements={num_elements}.")
        return None, None
        
    return x_nodes, U_fem

# --- Тестирование сходимости для МКЭ ---
print("\nТестирование сходимости МКЭ (f(x) = sin):")
N_elements_values = [4, 8, 16, 32, 64, 128, 256]
errors_L2_fem = []
errors_C_fem = []
h_values_fem = [] # h здесь это длина элемента

for N_el_test in N_elements_values:
    x_fem, u_fem_num = solve_1d_rod_fem(N_el_test, L, E, A, P, f_sin_load)
    if x_fem is None or u_fem_num is None:
        print(f"FEM: Решение для N_elements={N_el_test} не получено.")
        continue

    u_analyt_fem = analytical_solution_sin_load(x_fem, L, E, A, P, f_amplitude)
    
    error_l2_fem = np.sqrt(np.sum((u_fem_num - u_analyt_fem)**2) / (N_el_test + 1)) # (N+1) узлов
    errors_L2_fem.append(error_l2_fem)
    
    error_c_fem = np.max(np.abs(u_fem_num - u_analyt_fem))
    errors_C_fem.append(error_c_fem)
    
    current_h_fem = L / N_el_test # Характерный размер элемента
    h_values_fem.append(current_h_fem)
    print(f"N_el = {N_el_test:3d}, h_el = {current_h_fem:.4f}, Error (L2) = {error_l2_fem:.3e}, Error (C) = {error_c_fem:.3e}")

orders_L2_fem_calc = []
orders_C_fem_calc = []
if len(h_values_fem) > 1:
    for i in range(len(h_values_fem) - 1):
        p_l2 = (np.log(errors_L2_fem[i]) - np.log(errors_L2_fem[i+1])) / \
                 (np.log(h_values_fem[i]) - np.log(h_values_fem[i+1]))
        orders_L2_fem_calc.append(p_l2)
        p_c = (np.log(errors_C_fem[i]) - np.log(errors_C_fem[i+1])) / \
                (np.log(h_values_fem[i]) - np.log(h_values_fem[i+1]))
        orders_C_fem_calc.append(p_c)
    print("\nМКЭ Приблизительные порядки сходимости (L2):", [round(p, 2) for p in orders_L2_fem_calc])
    print("МКЭ Приблизительные порядки сходимости (C):", [round(p, 2) for p in orders_C_fem_calc])
else:
    print("\nМКЭ: Недостаточно данных для расчета порядка сходимости.")


# --- Графики для МКЭ ---
# 1. Сравнение МКЭ решения с аналитическим
N_el_plot = 32
x_fem_plot, u_fem_plot_num = solve_1d_rod_fem(N_el_plot, L, E, A, P, f_sin_load)
x_analyt_fine = np.linspace(0, L, 200) # Для гладкой аналитической кривой
u_analyt_fine = analytical_solution_sin_load(x_analyt_fine, L, E, A, P, f_amplitude)

if x_fem_plot is not None and u_fem_plot_num is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(x_fem_plot, u_fem_plot_num, 's', label=f'МКЭ (N_el={N_el_plot})', markersize=5, mfc='none', color='green')
    plt.plot(x_analyt_fine, u_analyt_fine, '--', label='Аналитическое', color='red', linewidth=1.5)
    plt.xlabel('Координата x, м')
    plt.ylabel('Перемещение u(x), м')
    plt.title(f'МКЭ: Сравнение численного и аналитического решения (N_el={N_el_plot})')
    plt.legend()
    plt.grid(True)
    plt.savefig('rod_displacement_comparison_fem.png')
    plt.show()

    # 2. График абсолютной ошибки для МКЭ
    u_analyt_on_fem_grid = analytical_solution_sin_load(x_fem_plot, L, E, A, P, f_amplitude)
    abs_error_fem_plot = np.abs(u_fem_plot_num - u_analyt_on_fem_grid)
    plt.figure(figsize=(10, 6))
    plt.plot(x_fem_plot, abs_error_fem_plot, '.-', label=f'МКЭ Абсолютная ошибка (N_el={N_el_plot})', color='purple')
    plt.xlabel('Координата x, м')
    plt.ylabel('Абсолютная ошибка |u_fem - u_analyt|, м')
    plt.title(f'МКЭ: Распределение абсолютной ошибки (N_el={N_el_plot})')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('rod_absolute_error_distribution_fem.png')
    plt.show()

# 3. График сходимости МКЭ
if h_values_fem and errors_L2_fem and errors_C_fem:
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values_fem, errors_L2_fem, 'o-', label='МКЭ Ошибка $L_2$', color='blue')
    plt.loglog(h_values_fem, errors_C_fem, 's--', label='МКЭ Ошибка $C$ (max)', color='cyan')
    
    h_theory_fem = np.array(h_values_fem)
    if errors_L2_fem:
        plt.loglog(h_theory_fem, errors_L2_fem[0] * (h_theory_fem/h_values_fem[0])**1, ':', color='gray', label='Теория $O(h)$')
        plt.loglog(h_theory_fem, errors_L2_fem[0] * (h_theory_fem/h_values_fem[0])**2, '--', color='black', label='Теория $O(h^2)$')

    plt.xlabel('Размер элемента $h_{el}$, м')
    plt.ylabel('Ошибка')
    plt.title('Сходимость метода конечных элементов (МКЭ)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.gca().invert_xaxis()
    plt.savefig('convergence_plot_fem.png')
    plt.show()
else:
    print("МКЭ: Не удалось построить график сходимости: нет данных.")