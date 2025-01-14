# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parâmetros físicos do sistema
g = 9.81  # Gravidade (m/s^2)
m1 = 1.0  # Massa do primeiro pêndulo (kg)
m2 = 1.0  # Massa do segundo pêndulo (kg)
l1 = 1.0  # Comprimento do primeiro pêndulo (m)
l2 = 1.0  # Comprimento do segundo pêndulo (m)
mu_env = 0.01  # Atrito no ambiente (kg/s)
mu_joint = 0.05  # Atrito nas junções (N*m*s/rad)

# Condições iniciais
def initialize_pendulum(theta1, theta2, omega1, omega2):
    return np.array([theta1, theta2, omega1, omega2])

# Equações do movimento
def equations(state, t):
    theta1, theta2, omega1, omega2 = state

    # Equações do pêndulo duplo com atrito
    delta_theta = theta2 - theta1

    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta_theta)**2
    denom2 = (l2 / l1) * denom1

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = (
        -g * (m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * m2 * l2 * omega2**2 * np.sin(delta_theta)
        - m2 * l1 * omega1**2 * np.sin(2 * delta_theta)
        - mu_env * omega1
        - mu_joint * omega1 / l1
    ) / denom1

    domega2_dt = (
        2 * l1 * omega1**2 * np.sin(delta_theta)
        + 2 * g * np.sin(theta1) * np.cos(delta_theta)
        + g * np.sin(theta2)
        + l1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta)
        - mu_env * omega2
        - mu_joint * omega2 / l2
    ) / denom2

    return [dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt]

# Método para resolver as EDOs
from scipy.integrate import odeint

def simulate_pendulum(state, t):
    return odeint(equations, state, t)

# Configurações da simulação
initial_state = initialize_pendulum(np.pi / 2, np.pi / 4, 0, 0)  # Condições iniciais
time = np.linspace(0, 200, 1000)  # Janela de tempo

# Resolver o sistema
solution = simulate_pendulum(initial_state, time)

# Conversão para coordenadas cartesianas
x1 = l1 * np.sin(solution[:, 0])
y1 = -l1 * np.cos(solution[:, 0])
x2 = x1 + l2 * np.sin(solution[:, 1])
y2 = y1 - l2 * np.cos(solution[:, 1])

# Função para atualizar a animação
def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    trace.set_data(x2[:frame], y2[:frame])
    return line, trace

# Configuração da animação
fig, ax = plt.subplots()
ax.set_xlim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_ylim(-2 * (l1 + l2), 2 * (l1 + l2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '-', lw=1, alpha=0.5)

ani = FuncAnimation(fig, update, frames=len(time), interval=30, blit=True)

# salvando como gif
ani.save('double_pendulum.gif', writer='pillow')

# Mostrar a animação
plt.show()

# %%
