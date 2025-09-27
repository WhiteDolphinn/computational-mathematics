import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_poisson(l, N, f, phi1, phi2, phi3, phi4):
    hx = l / (N - 1)
    hy = l / (N - 1)
    u = np.zeros((N, N))

    x = np.linspace(0, l, N)
    y = np.linspace(0, l, N)
    u[:, 0] = phi1(x)   
    u[:, -1] = phi2(x)  
    u[0, :] = phi3(y)   
    u[-1, :] = phi4(y)  

    # Преобразование f(x, y) в сеточный вид
    X, Y = np.meshgrid(x, y, indexing='ij')
    F = f(X, Y)

    # Итерационный метод Якоби
    tol = 1e-6
    max_iter = 10000
    for _ in range(max_iter):
        u_new = np.copy(u)
        for i in range(1, N-1):
            for j in range(1, N-1):
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - hx**2 * F[i, j])

        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            break
        u = u_new

    return X, Y, u, F

f = lambda x, y: np.sin(np.pi*2 * x) * np.sin(np.pi*3 * y)
phi1 = lambda x: np.zeros_like(x)
phi2 = lambda x: np.zeros_like(x)
phi3 = lambda y: np.zeros_like(y)
phi4 = lambda y: np.zeros_like(y)

l = 1
N = 50
X, Y, U, F = solve_poisson(l, N, f, phi1, phi2, phi3, phi4)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, U, levels=20, cmap='viridis')
plt.colorbar(contour, label='Значение u(x,y)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Численное решение уравнения Пуассона', fontsize=14)
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
plt.savefig("4-0.png", dpi=300, bbox_inches='tight')
# plt.show()


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
contour_f = plt.contourf(X, Y, F, levels=20, cmap='RdYlBu')
plt.colorbar(contour_f, label='Значение f(x,y)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Функция f(x,y)', fontsize=14)
plt.grid(alpha=0.3)

ax_f = plt.subplot(1, 2, 2, projection='3d')
surf_f = ax_f.plot_surface(X, Y, F, cmap='coolwarm', 
                         rstride=1, cstride=1,
                         linewidth=0, antialiased=True)
plt.colorbar(surf_f, ax=ax_f, label='Значение f(x,y)')
ax_f.set_xlabel('x', fontsize=12)
ax_f.set_ylabel('y', fontsize=12)
ax_f.set_zlabel('f(x,y)', fontsize=12)
ax_f.set_title('3D визуализация функции источника', fontsize=14)
ax_f.view_init(30, 45)

plt.tight_layout()
plt.savefig("4.png", dpi=300, bbox_inches='tight')
# plt.show()