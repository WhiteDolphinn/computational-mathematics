import numpy as np
import matplotlib.pyplot as plt

# --- Параметры задачи ---
L = 1.0  # Длина стержня, м
E = 2.1e11  # Модуль Юнга, Па (сталь)
A = 1e-4  # Площадь поперечного сечения, м^2
P = 1e4  # Сила на правом конце, Н
# Распределенная нагрузка (для простоты возьмем f(x) = const)
f_const = 1e3 # Н/м

# --- Аналитическое решение (для f(x) = f_const и P) ---
# Общее решение u'' = -f_const / (EA) => u(x) = -f_const/(2EA) * x^2 + C1*x + C2
# u(0) = 0 => C2 = 0
# EA u'(L) = P => EA * (-f_const/(EA)*L + C1) = P => -f_const*L + EA*C1 = P => C1 = (P + f_const*L) / (EA)
def analytical_solution(x, L, E, A, P, f_val):
    C1 = (P + f_val * L) / (E * A)
    C2 = 0
    return -f_val / (2 * E * A) * x**2 + C1 * x + C2

def analytical_solution_sin_load(x, L, E, A, P, f_amp):
    # f_amp - это амплитуда синусоидальной нагрузки (1e3 в нашем случае)
    C1 = (P + f_amp * L / np.pi) / (E * A)
    C2 = 0
    return (f_amp * L**2) / (E * A * np.pi**2) * np.sin(np.pi * x / L) + C1 * x + C2

# --- Численное решение (Метод конечных разностей + Метод прогонки) ---
def solve_1d_rod_mkr(N, L, E, A, P, f_func):
    h = L / N  # Шаг сетки
    x_nodes = np.linspace(0, L, N + 1)

    # Формирование СЛАУ Au = b
    # u_0, u_1, ..., u_N - всего N+1 неизвестное
    # Но u_0 известно, поэтому решаем для u_1, ..., u_N (N неизвестных)
    # Однако, для удобства прогонки, лучше строить систему для u_0...u_N
    # и потом модифицировать для граничных условий.
    # Либо сразу для N неизвестных u_1...u_N

    # Система для N неизвестных u_1, ..., u_N
    # Для i = 1, ..., N-1 (внутренние узлы):
    # u_{i-1} - 2u_i + u_{i+1} = -h^2/(EA) * f_i
    # Для i = N (правая граница):
    # EA * (3u_N - 4u_{N-1} + u_{N-2}) / (2h) = P
    # (3/(2h)) u_N - (4/(2h)) u_{N-1} + (1/(2h)) u_{N-2} = P/(EA)

    # Инициализируем матрицу A (N x N) и вектор b (N)
    # Индексы в матрице будут от 0 до N-1, соответствующие u_1 до u_N
    # u_vec[k] соответствует u_{k+1} в физических индексах
    
    mat_A = np.zeros((N, N))
    vec_b = np.zeros(N)

    # Заполнение для внутренних узлов (i = 1 to N-1, в матрице k = 0 to N-2)
    # u_i в уравнении это u_vec[i-1]
    for i_phys in range(1, N): # i_phys от 1 до N-1 (индекс в u)
        k_mat = i_phys - 1 # индекс строки в матрице (от 0 до N-2)
        
        f_i = f_func(x_nodes[i_phys])
        vec_b[k_mat] = -h**2 / (E * A) * f_i

        if i_phys == 1: # u_0 = 0
            mat_A[k_mat, k_mat] = -2  # Коэфф. при u_1 (u_vec[0])
            mat_A[k_mat, k_mat + 1] = 1    # Коэфф. при u_2 (u_vec[1])
            vec_b[k_mat] -= 0 # u_0 = 0 (переносим в правую часть)
        else: # i_phys > 1
            mat_A[k_mat, k_mat - 1] = 1    # Коэфф. при u_{i-1} (u_vec[i-2])
            mat_A[k_mat, k_mat] = -2  # Коэфф. при u_i (u_vec[i-1])
            if i_phys < N : # если не последний узел из этой группы
                 mat_A[k_mat, k_mat + 1] = 1    # Коэфф. при u_{i+1} (u_vec[i])

    # Заполнение для правого граничного условия (i_phys = N, k_mat = N-1)
    k_mat_N = N - 1
    # EA * (3u_N - 4u_{N-1} + u_{N-2}) / (2h) = P
    # (3)u_N - (4)u_{N-1} + (1)u_{N-2} = 2hP/(EA)
    # u_N это u_vec[N-1], u_{N-1} это u_vec[N-2], u_{N-2} это u_vec[N-3]
    
    if N >= 2: # Нужно хотя бы u_N и u_{N-1}
        mat_A[k_mat_N, k_mat_N]     = 3.0  # при u_N
        mat_A[k_mat_N, k_mat_N - 1] = -4.0 # при u_{N-1}
        if N >= 3: # Нужно u_{N-2}
            mat_A[k_mat_N, k_mat_N - 2] = 1.0  # при u_{N-2}
        elif N == 2: # Если всего N=2 (узлы u0, u1, u2), то u_{N-2} это u0=0
             pass # член с u0 переносится в правую часть, но он 0
        vec_b[k_mat_N] = (2 * h * P) / (E * A)
        
        # Если в предыдущем цикле для k_mat = N-2 (i_phys = N-1) был добавлен член u_{i+1} = u_N
        # он мог быть перезаписан. Проверим и восстановим, если надо.
        # Это происходит, если уравнение для u_{N-1} и уравнение для u_N используют u_{N-1} и u_{N-2}
        # Последнее уравнение (граничное) главнее для этих коэффициентов.
        # Проще всего переопределить последнюю строку и предпоследнюю строку для N-1, N-2
        # Если N=1 (только u1), то это уравнение не используется, так как нет u_{N-1}, u_{N-2}
        # Его нужно обрабатывать отдельно
        if N == 1: # Только узел u_1 (mat_A[0,0], vec_b[0])
            # u_0 - 2u_1 + u_2 = -h^2/(EA)f_1  --- это для внутреннего
            # Здесь N=1, значит L=h. Узлы x0=0, x1=L.
            # u0=0. Граничное условие на x1=L: EA u'(L) = P
            # Аппроксимация: EA (u1-u0)/h = P => EA u1/h = P => u1 = Ph/(EA)
            mat_A[0,0] = 1.0
            vec_b[0] = P*h/(E*A) # Если f=0

            # С учетом f(x):
            # Для N=1, узлы x0, x1. Уравнение для x1 (u_N):
            # EA (u1-u0)/h + f(x1)*h/2 = P  (интегральное осреднение нагрузки)
            # или EA (u1-u0)/h = P - f(x1)*h/2 (если f - объемная, то должна быть в диффуре)
            # Вернемся к дифф. уравнению: EA u'' + f = 0
            # EA (u_L - u_0)/L = P - integral(f,0,L)/L ?? нет.
            # Проще всего для N=1 использовать более простую аппроксимацию для du/dx
            # EA (u_N - u_{N-1})/h = P. Здесь u_N = u_1, u_{N-1} = u_0 = 0
            # EA u_1/h = P
            # А уравнение EA u'' + f = 0 тогда где?
            # Это сложный случай для такой аппроксимации.
            # Для очень грубых сеток лучше использовать вариационные методы (МКЭ).
            # Пока оставим как есть, для N=1 результат будет не очень.
            # Этот код не будет точен для N=1 с такой аппроксимацией ГУ.
            # Для N=1 лучше аналитически: u1 = (P*L + f_const*L^2/2)/(EA)
            pass


    # Решение СЛАУ
    try:
        u_solved_partial = np.linalg.solve(mat_A, vec_b)
    except np.linalg.LinAlgError:
        print(f"Singular matrix for N={N}. Check boundary conditions or problem setup.")
        print("Matrix A:\n", mat_A)
        print("Vector b:\n", vec_b)
        return None, None

    # Полный вектор решения, включая u_0
    u_full = np.zeros(N + 1)
    u_full[0] = 0  # u_0 = 0
    u_full[1:] = u_solved_partial

    return x_nodes, u_full

# --- Тестирование сходимости ---
N_values = [10, 20, 40, 80, 160] # Количество интервалов
errors_L2 = []
errors_C = []
h_values = []

# Создаем функцию для распределенной нагрузки
def f_load_function(x_coord):
    # return f_const # Постоянная нагрузка
    return 1e3 * np.sin(np.pi * x_coord / L) # Пример переменной нагрузки

# Для f(x) = f_const аналитическое решение есть.
# Если f_load_function не константа, аналитическое решение будет другим или его нет.
# Для теста сходимости лучше иметь аналитическое решение.
# Используем f(x) = f_const для теста сходимости.
def f_const_load(x_coord):
    return f_const

print("Тестирование сходимости (f(x) = const):")
for N_test in N_values:
    x_num, u_num = solve_1d_rod_mkr(N_test, L, E, A, P, f_const_load)
    if x_num is None:
        continue

    u_analyt = analytical_solution(x_num, L, E, A, P, f_const)
    
    # L2 норма ошибки
    error_l2 = np.sqrt(np.sum((u_num - u_analyt)**2) / (N_test + 1))
    errors_L2.append(error_l2)
    
    # C норма ошибки (максимальная абсолютная ошибка)
    error_c = np.max(np.abs(u_num - u_analyt))
    errors_C.append(error_c)
    
    h_values.append(L / N_test)
    print(f"N = {N_test:3d}, h = {L/N_test:.4f}, Error (L2) = {error_l2:.2e}, Error (C) = {error_c:.2e}")

# Оценка порядка сходимости (p)
# error = C * h^p  => log(error) = log(C) + p * log(h)
# p = (log(error1) - log(error2)) / (log(h1) - log(h2))
orders_L2 = []
orders_C = []
for i in range(len(h_values) - 1):
    p_l2 = (np.log(errors_L2[i]) - np.log(errors_L2[i+1])) / (np.log(h_values[i]) - np.log(h_values[i+1]))
    orders_L2.append(p_l2)
    p_c = (np.log(errors_C[i]) - np.log(errors_C[i+1])) / (np.log(h_values[i]) - np.log(h_values[i+1]))
    orders_C.append(p_c)

print("\nПриблизительные порядки сходимости (L2):", [round(p, 2) for p in orders_L2])
print("Приблизительные порядки сходимости (C):", [round(p, 2) for p in orders_C])
# Ожидаем порядок p=2, так как аппроксимация u'' и du/dx второго порядка.

# --- Построение графиков ---

# 1. График решения для одного N
N_plot = 40
x_plot, u_plot = solve_1d_rod_mkr(N_plot, L, E, A, P, f_const_load) # или f_load_function
x_analyt_fine = np.linspace(0, L, 200)
u_analyt_fine = analytical_solution(x_analyt_fine, L, E, A, P, f_const) # или f_const_load

plt.figure(figsize=(10, 6))
if x_plot is not None:
    plt.plot(x_plot, u_plot, 'o-', label=f'Численное решение (N={N_plot})', markersize=4)
plt.plot(x_analyt_fine, u_analyt_fine, '--', label='Аналитическое решение (f=const)', color='red')
plt.xlabel('Координата x, м')
plt.ylabel('Перемещение u(x), м')
plt.title('Перемещение вдоль стержня')
plt.legend()
plt.grid(True)
plt.savefig('rod_displacement.png') # Для вставки в курсовую
plt.show()

# 2. График сходимости (ошибка от шага сетки h)
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_L2, 'o-', label='Ошибка $L_2$')
plt.loglog(h_values, errors_C, 's--', label='Ошибка $C$ (max)')
# Теоретические линии для p=1 и p=2
if len(h_values) > 0 and len(errors_L2) > 0:
    h_theory = np.array(h_values)
    plt.loglog(h_theory, errors_L2[0] * (h_theory/h_values[0])**1, ':', color='gray', label='Теория $O(h)$')
    plt.loglog(h_theory, errors_L2[0] * (h_theory/h_values[0])**2, '--', color='black', label='Теория $O(h^2)$')

plt.xlabel('Шаг сетки h, м')
plt.ylabel('Ошибка')
plt.title('Сходимость численного метода')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.gca().invert_xaxis() # h уменьшается слева направо
plt.savefig('convergence_plot.png') # Для вставки в курсовую
plt.show()