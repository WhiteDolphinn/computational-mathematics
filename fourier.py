import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec

alpha = 0.01       
L = 1.0            
T = 1.0            
Nt = 1000          
dt = T / Nt        

def build_matrices(Nx, alpha, dt, dx):
    A = np.zeros((Nx, Nx))
    B = np.zeros((Nx, Nx))
    r = alpha * dt / (2 * dx**2)
    
    for i in range(1, Nx-1):
        A[i, i-1] = -r * 0.5
        A[i, i] = 1 + 2 * r * 0.5
        A[i, i+1] = -r * 0.5

        B[i, i-1] = r * (1 - 0.5)
        B[i, i] = 1 - 2 * r * (1 - 0.5)
        B[i, i+1] = r * (1 - 0.5)
    
    A[0, 0] = A[-1, -1] = 1
    B[0, 0] = B[-1, -1] = 1
    
    return A, B

def fourier_solution(x, t, u, N_terms=10):
    u_fourier = np.zeros_like(x)
    for n in range(1, N_terms + 1):
        lambda_n = 50*(n * np.pi / L)**2
        integral = (2 / L) * np.trapz(u * np.sin(n * np.pi * x / L), x)
        u_fourier += integral * np.sin(n * np.pi * x / L) * np.exp(-alpha**2 * lambda_n * t)
    return u_fourier


def compute_solutions(Nx):
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    
    # Численное решение
    A, B = build_matrices(Nx, alpha, dt, dx)
    u_num = np.sin(np.pi * x / L)  # Начальное условие
    
    for _ in range(Nt):
        u_num = np.linalg.solve(A, B @ u_num)
    
    # Аналитическое решение
    u_exact = np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi/L)**2 * T)
    
    return x, u_num, u_exact

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# График 1: Сравнение решений для различных Nx
ax1 = fig.add_subplot(gs[0, 0])
# Nx = 100
Nx_values = [10, 20, 50, 100, 200, 500, 1000]
for Nx in Nx_values:
    x, u_num, u_exact = compute_solutions(Nx)
    ax1.plot(x, u_num, '-', linewidth=2, label=f"Численное решение при Nx={Nx}")
ax1.plot(x, u_exact, '--', linewidth=2, label=f"Аналитическое решение")
ax1.set_title(f'Сравнение решений')
ax1.set_xlabel('x')
ax1.set_ylabel('u(x,T)')
ax1.legend()
ax1.grid(True)

# График 2: Зависимость погрешности от Nx
ax2 = fig.add_subplot(gs[0, 1])
Nx_values = [10, 20, 50, 100, 200, 500, 1000]
errors = []

for Nx in Nx_values:
    _, u_num, u_exact = compute_solutions(Nx)
    errors.append(np.max(np.abs(u_num - u_exact)))

log_Nx = np.log(Nx_values)
log_err = np.log(errors)
slope, intercept, r_value, _, _ = stats.linregress(log_Nx, log_err)

ax2.loglog(Nx_values, errors, 'bo-', label='Gогрешность')
ax2.loglog(Nx_values, np.exp(intercept)*np.array(Nx_values)**slope, 
         'r--', label=f'k={slope:.2f}')
ax2.set_title('Зависимость погрешности от числа узлов')
ax2.set_xlabel('Число узлов (Nx)')
ax2.set_ylabel('Максимальная погрешность')
ax2.legend()
ax2.grid(True, which="both", ls="-")

# График 3: Временная эволюция решения
ax3 = fig.add_subplot(gs[1, :])
Nx = 100
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
A, B = build_matrices(Nx, alpha, dt, dx)

u_evolution = np.sin(np.pi * x / L)
time_points = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
current_time = 0

for t in time_points[1:]:
    steps = int((t - current_time) / dt)
    for _ in range(steps):
        u_evolution = np.linalg.solve(A, B @ u_evolution)
    current_time = t
    ax3.plot(x, u_evolution, label=f't = {t:.1f}')

ax3.set_title('Эволюция численного решения')
ax3.set_xlabel('x')
ax3.set_ylabel('u(x,t)')
ax3.legend()
ax3.grid(True)

plt.figtext(0.15, 0.02, 
           f'Порядок сходимости: {slope:.2f}\n',
           bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('heat_equation_analysis.png', dpi=300)
plt.show()