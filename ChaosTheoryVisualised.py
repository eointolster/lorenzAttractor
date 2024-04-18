import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_lorenz_attractor(xs, ys, zs):
    """
    Plots the Lorenz Attractor in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor")
    plt.show()

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
x0, y0, z0 = 0.1, 0.0, 0.0

# Simulation settings
num_steps = 10000
dt = 0.01

# Simulate the Lorenz Attractor
xs, ys, zs = simulate_lorenz(num_steps, dt, sigma, rho, beta, x0, y0, z0)

# Plot the Lorenz Attractor
plot_lorenz_attractor(xs, ys, zs)