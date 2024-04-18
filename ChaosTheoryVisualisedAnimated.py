import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def lorenz(x, y, z, sigma, rho, beta):
    """
    Computes the derivatives of the Lorenz equations.
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def simulate_lorenz(num_steps, dt, sigma, rho, beta, x0, y0, z0):
    """
    Simulates the Lorenz Attractor using the Lorenz equations.
    """
    xs = np.zeros(num_steps + 1)
    ys = np.zeros(num_steps + 1)
    zs = np.zeros(num_steps + 1)

    xs[0], ys[0], zs[0] = x0, y0, z0

    for i in range(num_steps):
        dx, dy, dz = lorenz(xs[i], ys[i], zs[i], sigma, rho, beta)
        xs[i + 1] = xs[i] + dx * dt
        ys[i + 1] = ys[i] + dy * dt
        zs[i + 1] = zs[i] + dz * dt

    return xs, ys, zs

def animate_lorenz_attractor(num_steps, xs, ys, zs):
    """
    Animates the Lorenz Attractor in 3D with colors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))

    def update(i):
        ax.clear()
        ax.set_xlim([-20, 20])
        ax.set_ylim([-30, 30])
        ax.set_zlim([0, 50])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Lorenz Attractor")

        ax.plot(xs[:i], ys[:i], zs[:i], color=colors[i], lw=0.5)
        ax.scatter(xs[i], ys[i], zs[i], color=colors[i], marker='o', s=10)

    anim = FuncAnimation(fig, update, frames=num_steps, interval=10)
    plt.show()

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
x0, y0, z0 = 0.1, 0.0, 0.0

# Simulation settings
num_steps = 1000
dt = 0.01

# Simulate the Lorenz Attractor
xs, ys, zs = simulate_lorenz(num_steps, dt, sigma, rho, beta, x0, y0, z0)

# Animate the Lorenz Attractor
animate_lorenz_attractor(num_steps, xs, ys, zs)