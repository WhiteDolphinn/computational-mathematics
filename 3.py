import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

def initial_condition(x):
    return np.exp(-x**2)

def analytical_solution(x, t, a=1.0):
    """Аналитическое решение уравнения переноса для начального условия exp(-x^2)"""
    return np.exp(-(x - a*t)**2)

def solve_scheme(scheme, x, t_max, tau, h, a):
    Nx = len(x)
    Nt = int(t_max / tau)
    u = np.zeros((Nt + 1, Nx))
    u[0, :] = initial_condition(x)

    if scheme == "upwind":
        for n in range(Nt):
            u[n+1, 1:] = u[n, 1:] - (a * tau / h) * (u[n, 1:] - u[n, :-1])
            u[n+1, 0] = u[n, 0]
            
    elif scheme == "downwind":
        for n in range(Nt):
            u[n+1, :-1] = u[n, :-1] + (a * tau / h) * (u[n, 1:] - u[n, :-1])
            u[n+1, -1] = u[n, -1]
            
    elif scheme == "central":
        for n in range(Nt):
            u[n+1, 1:-1] = u[n, 1:-1] - (a * tau / (2*h)) * (u[n, 2:] - u[n, :-2])
            u[n+1, 0] = u[n, 0]
            u[n+1, -1] = u[n, -1]
            
    elif scheme == "lax":
        for n in range(Nt):
            u[n+1, 1:-1] = 0.5*(u[n, 2:] + u[n, :-2]) - (a * tau / (2*h)) * (u[n, 2:] - u[n, :-2])
            u[n+1, 0] = u[n, 0]
            u[n+1, -1] = u[n, -1]
            
    elif scheme == "lax_wendroff":
        c = a * tau / h
        for n in range(Nt):
            u[n+1, 1:-1] = (u[n, 1:-1] - 0.5*c*(u[n, 2:] - u[n, :-2]) + 
                            0.5*c**2*(u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
            u[n+1, 0] = u[n, 0]
            u[n+1, -1] = u[n, -1]

    return u

def calculate_errors(u_numerical, x, t, h, a=1.0):
    """Вычисление различных норм ошибки"""
    u_analytical = analytical_solution(x, t, a)
    error = u_numerical - u_analytical
    
    # L2 норма ошибки
    l2_error = np.sqrt(h * np.sum(error**2))
    
    # Максимальная ошибка
    max_error = np.max(np.abs(error))
    
    # Норма L1
    l1_error = h * np.sum(np.abs(error))
    
    return l2_error, max_error, l1_error

# Параметры исследования
x_min, x_max = -10, 10
a = 1.0
t_max = 4
schemes = {
    "upwind": "По потоку (downwind)",
    "downwind": "Против потока (upwind)",
    "central": "Центральная разность",
    "lax": "Схема Лакса",
    "lax_wendroff": "Лакса-Вендроффа"
}

# Исследование зависимости ошибок от времени
def time_error_analysis():
    Nx = 200
    h = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max, Nx)
    tau = 0.05
    times = np.linspace(0.1, t_max, 20)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle('Зависимость ошибок от времени при фиксированном шаге сетки', y=0.95)
    
    for i, (scheme_key, scheme_name) in enumerate(schemes.items()):
        l2_errors = []
        max_errors = []
        l1_errors = []
        
        for t in times:
            solution = solve_scheme(scheme_key, x, t, tau, h, a)
            u_numerical = solution[-1, :]
            l2_err, max_err, l1_err = calculate_errors(u_numerical, x, t, h, a)
            l2_errors.append(l2_err)
            max_errors.append(max_err)
            l1_errors.append(l1_err)
        
        plt.subplot(2, 3, i+1)
        plt.plot(times, l2_errors, 'o-', label='L2 ошибка')
        plt.plot(times, max_errors, 's--', label='Макс. ошибка')
        plt.plot(times, l1_errors, '^:', label='L1 ошибка')
        plt.title(scheme_name)
        plt.xlabel('Время')
        plt.ylabel('Ошибка')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('3-time-err.png')
    # plt.show()

# Исследование зависимости ошибок от шага сетки
def convergence_analysis():
    t_fixed = 1.0
    # Nxs = [100, 200, 400, 800, 1600]  # Увеличиваем диапазон сеток
    Nxs = [400, 800, 1600, 3200, 6400]
    schemes_params = {
        "upwind": {"cfl": 0.9, "order": -1},
        "downwind": {"cfl": 0.1, "order": 1},
        "central": {"cfl": 0.01, "order": -1},
        "lax": {"cfl": 0.8, "order": 2},
        "lax_wendroff": {"cfl": 0.8, "order": 2}
    }
    
    plt.figure(figsize=(12, 8))
    
    for scheme_key, scheme_name in schemes.items():
        l2_errors = []
        hs = []
        
        for Nx in Nxs:
            h = (x_max - x_min) / Nx
            tau = schemes_params[scheme_key]["cfl"] * h / a
            
            x = np.linspace(x_min, x_max, Nx)
            solution = solve_scheme(scheme_key, x, t_fixed, tau, h, a)
            u_numerical = solution[-1, :]
            u_analytical = analytical_solution(x, t_fixed, a)
            error = u_numerical - u_analytical
            # l2_error = np.sqrt(h * np.sum(error**2))
            l2_error = np.max(h*error)
            l2_errors.append(l2_error)
            hs.append(h)
        
        # Анализ только для устойчивых схем
        if schemes_params[scheme_key]["order"] > 0:
            log_h = np.log(np.array(hs))
            log_err = np.log(np.array(l2_errors))
            slope, intercept = np.polyfit(log_h, log_err, 1)
            
            # Теоретический порядок
            theoretical_order = schemes_params[scheme_key]["order"]
            
            plt.loglog(hs, l2_errors, 'o-', 
                     label=f'{scheme_name}\nНаблюдаемый порядок: {slope:.2f}\nТеоретический: {theoretical_order}')
        # else:
            # plt.loglog(hs, l2_errors, 'x--', 
            #          label=f'{scheme_name}\n(неустойчивая схема)')
        
        # plt.errorbar(log_h, log_err, 0, 0, 'o--', 
        #               label=f'{scheme_name}\nНаблюдаемый порядок: {slope:.2f}\nТеоретический: {theoretical_order}')

        print(f"\nСхема: {scheme_name}")
        print("Шаги h:", [f"{x:.4f}" for x in hs])
        print("Ошибки:", [f"{x:.6f}" for x in l2_errors])
    
    plt.title(f'Анализ сходимости (t={t_fixed})', fontsize=14)
    plt.xlabel('Шаг сетки h', fontsize=12)
    plt.ylabel('Ошибка', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('3-2-3.png', bbox_inches='tight', dpi=300)
    # plt.show()

# Основная визуализация решений и ошибок
def main_visualization():
    Nx = 200
    h = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max, Nx)
    tau = 0.05
    times = [0, 1, 2, 3, 4]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(18, 15), dpi=100)
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Словарь для хранения ошибок
    errors = {scheme: {"L2": [], "Max": [], "L1": [], "times": []} for scheme in schemes}

    for idx, (scheme_key, scheme_name) in enumerate(schemes.items()):
        ax = fig.add_subplot(gs[idx//2, idx%2])
        
        solution = solve_scheme(scheme_key, x, max(times), tau, h, a)
        
        for i, t in enumerate(times):
            time_idx = int(t / tau)
            u_numerical = solution[time_idx, :]
            ax.plot(x, u_numerical, 
                    color=colors[i], 
                    linewidth=2,
                    alpha=0.8,
                    label=f't = {t}')
            
            # Вычисление ошибок (кроме t=0)
            if t > 0:
                l2_err, max_err, l1_err = calculate_errors(u_numerical, x, t, a)
                errors[scheme_key]["L2"].append(l2_err)
                errors[scheme_key]["Max"].append(max_err)
                errors[scheme_key]["L1"].append(l1_err)
                errors[scheme_key]["times"].append(t)
        
        # Добавляем аналитическое решение для сравнения
        for i, t in enumerate(times):
            ax.plot(x, analytical_solution(x, t), '--', 
                   color=colors[i], linewidth=1, alpha=0.3)
        
        ax.set_title(scheme_name, fontsize=12, pad=10)
        ax.set_xlabel('Пространство (x)', fontsize=10)
        ax.set_ylabel('Амплитуда (u)', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if idx == 0:
            ax.legend(fontsize=9, framealpha=1)

    # График ошибок
    ax_errors = fig.add_subplot(gs[2, :])
    markers = ['o', 's', '^', 'D', 'v']

    for i, (scheme_key, scheme_name) in enumerate(schemes.items()):
        if errors[scheme_key]["times"]:
            ax_errors.plot(errors[scheme_key]["times"], errors[scheme_key]["L2"], 
                          marker=markers[i], linestyle='-', 
                          label=f'{scheme_name} (L2)')
            ax_errors.plot(errors[scheme_key]["times"], errors[scheme_key]["Max"], 
                          marker=markers[i], linestyle='--', 
                          label=f'{scheme_name} (Max)', alpha=0.6)

    ax_errors.set_title('Зависимость ошибок от времени', fontsize=12, pad=10)
    ax_errors.set_xlabel('Время', fontsize=10)
    ax_errors.set_ylabel('Ошибка', fontsize=10)
    ax_errors.grid(True, linestyle='--', alpha=0.6)
    ax_errors.legend(fontsize=9, ncol=2)
    ax_errors.set_yscale('log')

    plt.suptitle('Сравнение численных схем для уравнения переноса с анализом ошибок', 
                 fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig("3-3.png", bbox_inches='tight')
    # plt.show()

    # Вывод таблицы с ошибками
    print("\nТаблица ошибок в последний момент времени t=4:")
    print("{:<25} {:<15} {:<15} {:<15}".format("Схема", "L2 ошибка", "Макс. ошибка", "L1 ошибка"))
    for scheme_key in schemes:
        if errors[scheme_key]["times"]:
            last_idx = len(errors[scheme_key]["L2"]) - 1
            print("{:<25} {:<15.5f} {:<15.5f} {:<15.5f}".format(
                schemes[scheme_key],
                errors[scheme_key]["L2"][last_idx],
                errors[scheme_key]["Max"][last_idx],
                errors[scheme_key]["L1"][last_idx]))


# def time_step_convergence_analysis():
#     h_fixed = 0.01  # Фиксированный шаг по пространству
#     Nx = int((x_max - x_min) / h_fixed)
#     x = np.linspace(x_min, x_max, Nx)
    
#     # Отношения шагов по времени (τ = CFL * h / a)
#     CFL_ratios = [0.1, 0.05, 0.025, 0.0125, 0.00625]  
#     t_fixed = 1.0
    
#     schemes_params = {
#         "upwind": {"order": 1, "color": "blue"},
#         "lax": {"order": 1, "color": "green"},
#         "lax_wendroff": {"order": 2, "color": "red"}
#     }
    
#     plt.figure(figsize=(12, 8))
    
#     for scheme_key in schemes_params.keys():
#         l2_errors = []
#         taus = []
        
#         for CFL in CFL_ratios:
#             tau = CFL * h_fixed / a
#             solution = solve_scheme(scheme_key, x, t_fixed, tau, h_fixed, a)
#             u_numerical = solution[-1, :]
#             u_analytical = analytical_solution(x, t_fixed, a)
            
#             # Исключаем граничные точки
#             interior = slice(Nx//4, 3*Nx//4)
#             error = u_numerical[interior] - u_analytical[interior]
#             l2_error = np.max(h_fixed * error)
            
#             l2_errors.append(l2_error)
#             taus.append(tau)
        
#         # Линейная регрессия в логарифмических координатах
#         log_tau = np.log(np.array(taus))
#         log_err = np.log(np.array(l2_errors))
#         slope, intercept = np.polyfit(log_tau, log_err, 1)
        
#         plt.loglog(taus, l2_errors, 'o-', 
#                  color=schemes_params[scheme_key]["color"],
#                  label=f'{schemes[scheme_key]}\nНаблюдаемый порядок: {slope:.2f}\nТеоретический: {schemes_params[scheme_key]["order"]}')
        
#         print(f"\n{schemes[scheme_key]}:")
#         print(f"τ = {taus}")
#         print(f"L2 ошибки = {l2_errors}")

#     # Теоретические линии сходимости
#     x_ref = np.array([min(taus), max(taus)])
#     plt.loglog(x_ref, 0.1*x_ref**1, 'k--', label='Теор. порядок 1')
#     plt.loglog(x_ref, 0.1*x_ref**2, 'k:', label='Теор. порядок 2')
    
#     plt.title(f'Анализ сходимости по времени (h={h_fixed}, t={t_fixed})', fontsize=14)
#     plt.xlabel('Шаг по времени τ', fontsize=12)
#     plt.ylabel('L2 ошибка', fontsize=12)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig('3-4.png', bbox_inches='tight', dpi=300)
#     # plt.show()

# Запуск всех исследований
if __name__ == "__main__":
    main_visualization()
    time_error_analysis()
    convergence_analysis()
    # time_step_convergence_analysis()