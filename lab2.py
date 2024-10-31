import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

m = 0.1
с = 5
L1 = 1 
L2 = 0.5
g = 9.81
phi, psi = np.radians(36), np.radians(36)
dphi, dpsi = 0, 0

dt = 0.1
timesteps = 100


time = np.arange(0, timesteps*dt, dt)
x1_values = np.zeros(timesteps)
y1_values = np.zeros(timesteps)
x2_values = np.zeros(timesteps)
y2_values = np.zeros(timesteps)

fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1.5)

M1 = plt.Circle((0, 1), 0.1, color='b', fill=True)
ax.add_patch(M1)
M2 = plt.Circle((0, 0), 0.1, color='b', fill=True)
ax.add_patch(M2)

line1, = plt.plot([], [], 'k')
line2, = plt.plot([], [], 'k')
line3 = plt.vlines(x=[-1, -1.2], ymin=-0.5, ymax=1.5, color='k')

rectangle = plt.Rectangle((-1.19, 0.85), 0.15, 0.3, color='b')
ax.add_patch(rectangle)


n = 50
h = 0.04
xP = np.linspace(0, 0, 2 * n + 1)

yP = h * np.sin(np.pi / 2 * np.arange(2 * n + 1))

spring, = plt.plot(xP, yP, 'k')

def update(frame):
    global phi, psi, dphi, dpsi, M1, M2, line1, line2, x1_values, y1_values, x2_values, y2_values
    alpha1 = -g / L1 * np.sin(phi)
    alpha2 = -g / L2 * np.sin(psi)
    dphi += alpha1 * dt
    dpsi += alpha2 * dt
    phi += dphi * dt
    psi += dpsi * dt

    x1 = 0 + L1 * np.sin(phi)
    y1 = -L1 * np.cos(phi) + 1
    x2 = x1 + L2 * np.sin(psi)
    y2 = y1 - L2 * np.cos(psi)

    # Обновление координат M1 и M2
    M1.center = (x1, y1)
    M2.center = (x2, y2)

    # Обновление линий соединения
    line1.set_data([0, x1], [1, y1])
    line2.set_data([x1, x2], [y1, y2])

    # Обновление груза
    rectangle.set_xy((-1.19, y1 - 0.15))

    # Обновление пружины
    xP = np.linspace(0 - 1, x1, 2 * n + 1)
    spring.set_data(xP, yP + y1)

    x1_values[frame] = x1
    y1_values[frame] = y1
    x2_values[frame] = x2
    y2_values[frame] = y2

    return M1, M2, line1, line2, rectangle, spring

animation = FuncAnimation(fig, update, frames=timesteps, interval=50)