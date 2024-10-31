import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def SystDiffEq(y, t, m, L1, L2, c, g):
    # y = [phi, psi, phi', psi'] -> dy = [phi', psi', phi'', psi'']
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    phi = y[0]
    psi = y[1]
    dphi = y[2]
    dpsi = y[3]

    # a11 * phi'' + a12 * psi'' = b1
    # a21 * phi'' + a22 * psi'' = b2

    a11 = 2 * L1
    a12 = L2 * np.cos(psi - phi)
    b1 = L2 * dpsi ** 2 * np.sin(psi - phi) - (2 * g + (c * L1 / m) * np.cos(phi)) * np.sin(phi)
    
    a21 = L1 * np.cos(psi - phi)
    a22 = L2
    b2 = - dphi ** 2 * np.sin(psi - phi) - g * np.sin(psi)

    detA = a11 * a22 - a12 * a21
    detA1 = b1 * a22 - a12 * b2
    detA2 = a11 * b2 - b1 * a21

    dy[2] = detA1 / detA
    dy[3] = detA2 / detA

    return dy
    
# Дано:
L1 = 0.5
L2 = 0.5
g = 9.81
m = 0.1
c = 500
t0 = 0
phi0 = np.pi / 10
psi0 = np.pi / 10
dphi0 = 0
dpsi0 = 0

# Задаю функции phi(t) и psi(t) 

steps = 1000

t = np.linspace(0, 10, steps)

y0 = np.array([phi0, psi0, dphi0, dpsi0])

Y = odeint(SystDiffEq, y0, t, (m, L1, L2, c, g))

phi = Y[:,0]
psi = Y[:,1]
dphi = Y[:,2]
dpsi = Y[:,3]

ddpsi = np.zeros_like(t)
for i in np.arange(len(t)):
    ddpsi[i] = SystDiffEq(Y[i], t[i], m, L1, L2, c, g)[2]

N = m * (g * np.cos(psi) - L1 * ddpsi * np.sin(psi - phi) + L1 * dphi**2 * np.cos(psi - phi) + dpsi**2 * L2)

fgrt = plt.figure()
phiplt = fgrt.add_subplot(3, 1, 1)
#plt.title("$\phi(t)$")
phiplt.set_xlabel('$t$')
phiplt.set_ylabel('$\phi(t)$')
phiplt.plot(t, phi, color = 'r')
psiplt = fgrt.add_subplot(3, 1, 2)
#plt.title("$\psi(t)$")
psiplt.set_xlabel('$t$')
psiplt.set_ylabel('$\psi(t)$')
psiplt.plot(t, psi, color = 'g')
nplt = fgrt.add_subplot(3, 1, 3)
#plt.title("N(t)")
nplt.set_xlabel('$t$')
nplt.set_ylabel('$N(t)$')
nplt.plot(t, N)


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
    global phi, psi, dphi, dpsi, M1, M2, line1, line2
    phi = Y[frame, 0]
    psi = Y[frame, 1]
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

    return M1, M2, line1, line2, rectangle, spring

animation = FuncAnimation(fig, update, frames=steps, interval=50)