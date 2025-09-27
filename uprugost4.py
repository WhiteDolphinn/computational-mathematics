import numpy as np
import matplotlib.pyplot as plt

# --- Параметры задачи и функции (оставляем как есть) ---
L = 100.0
E = 2.1e11
A = 1e-4
P = 1e4
f_amplitude = 1e3

def f_sin_load(x_coord):
    return f_amplitude * np.sin(np.pi * x_coord / L)

def analytical_solution_sin_load(x, L, E, A, P, f_amp):
    C1 = (P + f_amp * L / np.pi) / (E * A)
    C2 = 0
    u_particular_due_to_f = (f_amp * L**2) / (E * A * np.pi**2) * np.sin(np.pi * x / L)
    u_general_homogeneous = C1 * x + C2
    return u_particular_due_to_f + u_general_homogeneous

# --- Функция решения МКР (solve_1d_rod_mkr - оставляем как есть) ---
def solve_1d_rod_mkr(N_intervals, L, E, A, P_force, f_func):
    if N_intervals < 2: # Гарантируем минимум для ГУ 2-го порядка
        # print(f"МКР: N_intervals={N_intervals} слишком мало, используется N_intervals=2.")
        N_intervals = 2
        
    h = L / N_intervals
    x_nodes = np.linspace(0, L, N_intervals + 1)
    mat_A = np.zeros((N_intervals, N_intervals))
    vec_b = np.zeros(N_intervals)

    for i_phys in range(1, N_intervals): 
        k_mat = i_phys - 1
        f_i = f_func(x_nodes[i_phys])
        vec_b[k_mat] = -h**2 / (E * A) * f_i
        if i_phys == 1: 
            mat_A[k_mat, k_mat] = -2
            if N_intervals > 1: mat_A[k_mat, k_mat + 1] = 1
        else: 
            mat_A[k_mat, k_mat - 1] = 1
            mat_A[k_mat, k_mat] = -2
            if i_phys < N_intervals: mat_A[k_mat, k_mat + 1] = 1
    
    if N_intervals > 0:
        k_mat_N = N_intervals - 1
        if k_mat_N < mat_A.shape[0]: # Проверка индекса
            mat_A[k_mat_N, k_mat_N]     = 3.0
            if N_intervals >= 2: mat_A[k_mat_N, k_mat_N - 1] = -4.0
            if N_intervals >= 3: mat_A[k_mat_N, k_mat_N - 2] = 1.0
            vec_b[k_mat_N] = (2 * h * P_force) / (E * A) # Перезаписываем b для ГУ
        
    if N_intervals == 0: return x_nodes, np.array([0.0])

    try:
        u_solved_partial = np.linalg.solve(mat_A, vec_b)
    except np.linalg.LinAlgError:
        print(f"МКР: Singular matrix for N_intervals={N_intervals}.")
        return None, None

    u_full = np.zeros(N_intervals + 1)
    u_full[0] = 0
    if N_intervals > 0: u_full[1:] = u_solved_partial
    return x_nodes, u_full

# --- Функция решения МКЭ (solve_1d_rod_fem - оставляем как есть) ---
def solve_1d_rod_fem(num_elements, L, E, A, P_force, f_func):
    if num_elements < 1: # Минимум 1 элемент
        # print(f"МКЭ: num_elements={num_elements} слишком мало, используется num_elements=1.")
        num_elements = 1

    num_nodes = num_elements + 1
    x_nodes = np.linspace(0, L, num_nodes)
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)

    for e in range(num_elements):
        node1_idx, node2_idx = e, e + 1
        x1, x2 = x_nodes[node1_idx], x_nodes[node2_idx]
        h_e = x2 - x1
        k_e_local = (E * A / h_e) * np.array([[1, -1], [-1, 1]])
        indices = np.array([node1_idx, node2_idx])
        K_global[np.ix_(indices, indices)] += k_e_local
        f_at_node1, f_at_node2 = f_func(x1), f_func(x2)
        f_e_local_contrib = (h_e / 6.0) * np.array([2 * f_at_node1 + f_at_node2,
                                                   f_at_node1 + 2 * f_at_node2])
        F_global[indices] += f_e_local_contrib
    F_global[num_nodes - 1] += P_force
    K_modified = np.copy(K_global); F_modified = np.copy(F_global)
    K_modified[0, :] = 0.0; K_modified[:, 0] = 0.0
    K_modified[0, 0] = 1.0; F_modified[0] = 0.0
    try:
        U_fem = np.linalg.solve(K_modified, F_modified)
    except np.linalg.LinAlgError:
        print(f"МКЭ: Singular matrix for num_elements={num_elements}.")
        return None, None
    return x_nodes, U_fem

# --- Расчет ошибок для МКР ---
print("Тестирование сходимости МКР (f(x) = sin):")
N_values_mkr = [4, 8, 16, 32, 64, 128, 256] # Число интервалов
errors_L2_mkr, errors_C_mkr, h_values_mkr = [], [], []
orders_L2_mkr_calc, orders_C_mkr_calc = [], []

for N_test in N_values_mkr:
    x_num, u_num = solve_1d_rod_mkr(N_test, L, E, A, P, f_sin_load)
    if x_num is None: continue
    u_analyt = analytical_solution_sin_load(x_num, L, E, A, P, f_amplitude)
    errors_L2_mkr.append(np.sqrt(np.sum((u_num - u_analyt)**2) / (N_test + 1)))
    errors_C_mkr.append(np.max(np.abs(u_num - u_analyt)))
    h_values_mkr.append(L / N_test)
    print(f"МКР: N_int = {N_test:3d}, h = {L/N_test:.4f}, Err(L2) = {errors_L2_mkr[-1]:.3e}, Err(C) = {errors_C_mkr[-1]:.3e}")

if len(h_values_mkr) > 1:
    for i in range(len(h_values_mkr) - 1):
        orders_L2_mkr_calc.append((np.log(errors_L2_mkr[i]) - np.log(errors_L2_mkr[i+1])) / (np.log(h_values_mkr[i]) - np.log(h_values_mkr[i+1])))
        orders_C_mkr_calc.append((np.log(errors_C_mkr[i]) - np.log(errors_C_mkr[i+1])) / (np.log(h_values_mkr[i]) - np.log(h_values_mkr[i+1])))
    print("МКР Приблизительные порядки сходимости (L2):", [round(p, 2) for p in orders_L2_mkr_calc])
    print("МКР Приблизительные порядки сходимости (C):", [round(p, 2) for p in orders_C_mkr_calc])

# --- Расчет ошибок для МКЭ ---
print("\nТестирование сходимости МКЭ (f(x) = sin):")
N_values_fem = [4, 8, 16, 32, 64, 128, 256] # Число элементов
errors_L2_fem, errors_C_fem, h_values_fem = [], [], []
orders_L2_fem_calc, orders_C_fem_calc = [], []

for N_el_test in N_values_fem:
    x_fem, u_fem_num = solve_1d_rod_fem(N_el_test, L, E, A, P, f_sin_load)
    if x_fem is None: continue
    u_analyt_fem = analytical_solution_sin_load(x_fem, L, E, A, P, f_amplitude)
    errors_L2_fem.append(np.sqrt(np.sum((u_fem_num - u_analyt_fem)**2) / (N_el_test + 1)))
    errors_C_fem.append(np.max(np.abs(u_fem_num - u_analyt_fem)))
    h_values_fem.append(L / N_el_test)
    print(f"МКЭ: N_el = {N_el_test:3d}, h_el = {L/N_el_test:.4f}, Err(L2) = {errors_L2_fem[-1]:.3e}, Err(C) = {errors_C_fem[-1]:.3e}")

if len(h_values_fem) > 1:
    for i in range(len(h_values_fem) - 1):
        orders_L2_fem_calc.append((np.log(errors_L2_fem[i]) - np.log(errors_L2_fem[i+1])) / (np.log(h_values_fem[i]) - np.log(h_values_fem[i+1])))
        orders_C_fem_calc.append((np.log(errors_C_fem[i]) - np.log(errors_C_fem[i+1])) / (np.log(h_values_fem[i]) - np.log(h_values_fem[i+1])))
    print("МКЭ Приблизительные порядки сходимости (L2):", [round(p, 2) for p in orders_L2_fem_calc])
    print("МКЭ Приблизительные порядки сходимости (C):", [round(p, 2) for p in orders_C_fem_calc])


# --- Общий график сходимости для МКР и МКЭ ---
plt.figure(figsize=(12, 7))

if h_values_mkr and errors_L2_mkr:
    plt.loglog(h_values_mkr, errors_L2_mkr, 'o-', label='МКР Ошибка $L_2$', color='blue', mfc='white')
if h_values_mkr and errors_C_mkr:
    plt.loglog(h_values_mkr, errors_C_mkr, 's--', label='МКР Ошибка $C$ (max)', color='cyan', mfc='white')

if h_values_fem and errors_L2_fem:
    plt.loglog(h_values_fem, errors_L2_fem, '^-', label='МКЭ Ошибка $L_2$', color='green', mfc='white')
if h_values_fem and errors_C_fem:
    plt.loglog(h_values_fem, errors_C_fem, 'd--', label='МКЭ Ошибка $C$ (max)', color='lime', mfc='white')

# Теоретические линии (используем данные МКР или МКЭ для начальной точки, они должны быть похожи)
if h_values_mkr and errors_L2_mkr: # Можно взять любую из серий данных для референса
    h_theory_ref = np.array(h_values_mkr) # или h_values_fem
    error_ref_L2 = errors_L2_mkr[0]     # или errors_L2_fem[0]
    plt.loglog(h_theory_ref, error_ref_L2 * (h_theory_ref/h_theory_ref[0])**1, ':', color='gray', label='Теория $O(h)$')
    plt.loglog(h_theory_ref, error_ref_L2 * (h_theory_ref/h_theory_ref[0])**2, '--', color='black', label='Теория $O(h^2)$')

plt.xlabel('Шаг сетки h (МКР) / Размер элемента $h_{el}$ (МКЭ), м')
plt.ylabel('Ошибка')
plt.title('Сравнение сходимости МКР и МКЭ')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.gca().invert_xaxis()
plt.savefig('convergence_plot_mkr_vs_fem.png')
plt.show()

# Остальные графики сравнения решений и разницы можно оставить как были, если нужны.
# Например, график сравнения решений для одного N:
N_compare = 32
x_mkr_comp, u_mkr_comp = solve_1d_rod_mkr(N_compare, L, E, A, P, f_sin_load)
x_fem_comp, u_fem_comp = solve_1d_rod_fem(N_compare, L, E, A, P, f_sin_load)
x_analyt_fine = np.linspace(0, L, 200)
u_analyt_fine = analytical_solution_sin_load(x_analyt_fine, L, E, A, P, f_amplitude)

plt.figure(figsize=(12, 7))
if x_mkr_comp is not None: plt.plot(x_mkr_comp, u_mkr_comp, 'o--', label=f'МКР (N={N_compare})', markersize=5, mfc='white', color='blue')
if x_fem_comp is not None: plt.plot(x_fem_comp, u_fem_comp, 's:', label=f'МКЭ (N_el={N_compare})', markersize=5, mfc='white', color='green')
plt.plot(x_analyt_fine, u_analyt_fine, '-', label='Аналитическое', color='red', linewidth=1.5)
plt.xlabel('Координата x, м'); plt.ylabel('Перемещение u(x), м')
plt.title(f'Сравнение решений МКР, МКЭ и Аналитического (N={N_compare})')
plt.legend(); plt.grid(True)
plt.savefig('rod_solutions_comparison_mkr_fem_detail.png') # новое имя файла
plt.show()