import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_poisson(l, N1, N2, f, phi1, phi2, phi3, phi4):
    hx = l / (N1 - 1)
    hy = l / (N2 - 1)
    u = np.zeros((N1, N2))

    x = np.linspace(0, l, N1)
    y = np.linspace(0, l, N2)
    u[:, 0] = phi1(x)   
    u[:, -1] = phi2(x)  
    u[0, :] = phi3(y)   
    u[-1, :] = phi4(y)  

    X, Y = np.meshgrid(x, y, indexing='ij')
    F = f(X, Y)

    tol = 1e-6
    max_iter = 10000
    for _ in range(max_iter):
        u_new = np.copy(u)
        for i in range(1, N1-1):
            for j in range(1, N2-1):
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - hx**2 * F[i, j])

        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            break
        u = u_new

    return X, Y, u, F

def exact_solution(x, y):
    return -np.sin(2*np.pi*x) * np.sin(3*np.pi*y) / (13*np.pi**2)

# Параметры задачи
f = lambda x, y: np.sin(2*np.pi*x) * np.sin(3*np.pi*y)
phi1 = lambda x: np.zeros_like(x)
phi2 = lambda x: np.zeros_like(x)
phi3 = lambda y: np.zeros_like(y)
phi4 = lambda y: np.zeros_like(y)
l = 1

# Список различных N для анализа сходимости
N_list = [5, 10, 20, 30, 40, 50, 60, 80]
errors = []
h_list = []

for N in N_list:
    X, Y, U, F = solve_poisson(l, N, 200, f, phi1, phi2, phi3, phi4)
    
    # Вычисляем аналитическое решение на текущей сетке
    U_exact = exact_solution(X, Y)
    
    # Вычисляем максимальную погрешность
    error = np.max(np.abs(U - U_exact))
    errors.append(error)
    h_list.append(l/(N-1))

# Построение графиков решения для последнего N (самой мелкой сетки)
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, U, levels=20, cmap='viridis')
plt.colorbar(contour, label='Значение u(x,y)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Численное решение (N={N})', fontsize=14)
plt.grid(alpha=0.3)

ax = plt.subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, U, cmap='plasma', 
                      rstride=1, cstride=1, 
                      linewidth=0, antialiased=True)
plt.colorbar(surf, ax=ax, label='Значение u(x,y)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('u(x,y)', fontsize=12)
ax.set_title('3D визуализация решения', fontsize=14)
ax.view_init(30, 45)

plt.tight_layout()
plt.savefig("poisson_solution.png", dpi=300, bbox_inches='tight')

log_h = np.log(np.array(h_list))
log_err = np.log(np.array(errors))
slope, intercept = np.polyfit(log_h, log_err, 1)

# График погрешности
plt.figure(figsize=(10, 6))
plt.loglog(h_list, errors, 'o-', label=f'Максимальная погрешность, порядок={slope}')
# plt.loglog(h_list, [h**2 for h in h_list], '--', label='O(h²)')

# Добавляем подписи к точкам
# for i, (h, err) in enumerate(zip(h_list, errors)):
#     if i % 2 == 0:  # Подписываем через одну для читаемости
#         plt.text(h, err, f'N={N_list[i]}', ha='center', va='bottom')

plt.xlabel('Шаг сетки h', fontsize=12)
plt.ylabel('Максимальная погрешность', fontsize=12)
plt.title('Зависимость погрешности от шага сетки', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("convergence2.png", dpi=300, bbox_inches='tight')

plt.show()