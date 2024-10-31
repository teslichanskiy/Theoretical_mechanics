import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Rot2D(x, y, a): # функция вращения стрелки
    Rx = x * np.cos(a) - y * np.sin(a)
    Ry = x * np.sin(a) + y * np.cos(a)
    return Rx, Ry

T = np.linspace(0, 10, 1000)
t = sp.Symbol('t')

r = 2 + sp.sin(12*t)
phi = t + 0.2 * sp.cos(13 * t)

X = np.zeros_like(T)
Y = np.zeros_like(T)
x = r * sp.cos(phi)
y = r * sp.sin(phi)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)
F_Ax = sp.lambdify(t, Ax)
F_Ay = sp.lambdify(t, Ay)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = F_x(T[i]) 
    Y[i] = F_y(T[i]) 
    VX[i] = F_Vx(T[i])
    VY[i] = F_Vy(T[i])
    AX[i] = F_Ax(T[i])
    AY[i] = F_Ay(T[i])
 

fig = plt.figure(figsize=[10,8])
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.grid(True)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set(xlim = [-5, 5], ylim = [-4, 4])
ax1.plot(X, Y)
VX , VY , AX, AY = 0.1*VX, 0.1*VY, 0.01*AX, 0.01*AY
P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r') 
WLine, = ax1.plot([X[0], X[0]+AX[0]], [Y[0], Y[0]+AY[0]], 'g') 
RLine, = ax1.plot([0, X[0]], [0, Y[0]], 'b') 
ArrowX = np.array([-0.05, 0, -0.05]) 
ArrowY = np.array([0.05, 0, -0.05])
VArrowX, VArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0])) 
VArrow, = ax1.plot(VArrowX+X[0]+VX[0], VArrowY+Y[0]+VY[0], 'r') 
WArrowX, WArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0])) 
WArrow, = ax1.plot(WArrowX+X[0]+AX[0], WArrowY+Y[0]+AY[0], 'g')
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0])) 
RArrow, = ax1.plot(RArrowX+X[0], RArrowY+Y[0], 'b') 


def animate(i):
    P.set_data(X[i], Y[i]) #координаты точки в каждый момент времени
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    VArrowX, VArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(ArrowX + X[i] + VX[i], ArrowY + Y[i] + VY[i])
    WLine.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    WArrowX, WArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    WArrow.set_data(WArrowX + X[i] + AX[i], WArrowY + Y[i] + AY[i])
    RLine.set_data([0, X[i]], [0, Y[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RArrowX + X[i], RArrowY + Y[i])
    return P, VLine, VArrow, WLine, WArrow, RLine, RArrow

anim = FuncAnimation(fig, animate, frames = len(T), interval = 50)
plt.show()