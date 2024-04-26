import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

def simple_pendulum(theta, t, L, b, g):
    """
    Defines the differential equation for the simple pendulum system.

    Parameters:
        theta: angle of the pendulum (radians)
        t: time
        L: length of the pendulum (m)
        b: damping coefficient (s^-1)
        g: acceleration due to gravity (m/s^2)

    Returns:
        dtheta_dt: angular velocity of the pendulum
    """
    dtheta_dt = theta[1]
    d2theta_dt2 = -(b * theta[1] + g * np.sin(theta[0])) / L
    return [dtheta_dt, d2theta_dt2]

def plot_pendulum(theta, t):
    """
    Plots the position of the pendulum vs. time.

    Parameters:
        theta: angle of the pendulum (degrees)
        t: time
    """
    plt.plot(t, np.degrees(theta[:, 0]))  # Convert angle from radians to degrees
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Simple Pendulum Motion')
    plt.grid(True)
    plt.show()

def update(frame, line, pendulum, L, b, g):
    """
    Update the animation frame.

    Parameters:
        frame: current frame number
        line: plot object for the pendulum
        pendulum: state of the pendulum (angle, angular velocity)
        L: length of the pendulum (m)
        b: damping coefficient (s^-1)
        g: acceleration due to gravity (m/s^2)

    Returns:
        line: updated plot object
    """
    theta = pendulum[frame, 0]
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    line.set_data([0, x], [0, y])
    return line,

# Parameters
L = 1.0  # Length of pendulum (m)
b = 0.25  # Damping coefficient (s^-1)
g = 9.81  # Acceleration due to gravity (m/s^2)
theta0 = np.radians(90.0)  # Initial angle (radians)
omega0 = 0.0  # Initial angular velocity (radians/s)
t = np.linspace(0, 10, 1000)  # Time array

# Solve the differential equation
pendulum = odeint(simple_pendulum, [theta0, omega0], t, args=(L, b, g))

# Plot the pendulum motion
plot_pendulum(pendulum, t)

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect('equal')
ax.grid(True)
line, = ax.plot([], [], 'o-', lw=2)

ani = FuncAnimation(fig, update, frames=len(t), fargs=(line, pendulum, L, b, g), blit=True, interval=10)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Pendulum Motion Animation')
plt.show()