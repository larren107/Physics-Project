import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def double_pendulum(t, state, L1, L2, m1, m2, g, b1, b2):
    """
    Defines the differential equations for the damped double pendulum system.

    Parameters:
        t: time
        state: vector of the system's state [theta1, omega1, theta2, omega2]
        L1, L2: lengths of the pendulums
        m1, m2: masses of the pendulums
        g: acceleration due to gravity
        b1, b2: damping coefficients for the pendulums

    Returns:
        state_dot: derivatives of the state variables [omega1, alpha1, omega2, alpha2]
    """
    theta1, omega1, theta2, omega2 = state

    delta_theta = theta2 - theta1
    delta_omega = omega2 - omega1

    # Equations of motion
    alpha1 = (m2 * g * np.sin(theta2) * np.cos(delta_theta) - m2 * np.sin(delta_theta) *
              (L1 * omega1 ** 2 * np.cos(delta_theta) + L2 * omega2 ** 2) -
              (m1 + m2) * g * np.sin(theta1) - b1 * omega1) / (L1 * (m1 + m2 * np.sin(delta_theta) ** 2))

    alpha2 = ((m1 + m2) * (L1 * omega1 ** 2 * np.sin(delta_theta) - g * np.sin(theta2) +
              g * np.sin(theta1) * np.cos(delta_theta)) + m2 * L2 * omega2 ** 2 * np.sin(delta_theta) *
              np.cos(delta_theta) - b2 * omega2) / (L2 * (m1 + m2 * np.sin(delta_theta) ** 2))

    return [omega1, alpha1, omega2, alpha2]

def update(frame, line1, line2, pendulum1, pendulum2, L1, L2):
    """
    Update the animation frame.

    Parameters:
        frame: current frame number
        line1: plot object for the first pendulum
        line2: plot object for the second pendulum
        pendulum1: state of the first pendulum (theta1, omega1, theta2, omega2)
        pendulum2: state of the second pendulum (theta1, omega1, theta2, omega2)
        L1, L2: lengths of the pendulums

    Returns:
        line1: updated plot object for the first pendulum
        line2: updated plot object for the second pendulum
    """
    x1 = L1 * np.sin(pendulum1[frame])
    y1 = -L1 * np.cos(pendulum1[frame])

    x2 = x1 + L2 * np.sin(pendulum2[frame])
    y2 = y1 - L2 * np.cos(pendulum2[frame])

    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])

    return line1, line2

def plot_double_pendulum(t, theta1, theta2):
    """
    Plots the angles of the double pendulum vs. time.

    Parameters:
        t: time
        theta1: angle of the first pendulum
        theta2: angle of the second pendulum
    """
    plt.plot(t, theta1, label='Theta 1')
    plt.plot(t, theta2, label='Theta 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radians)')
    plt.title('Double Pendulum Motion')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
L1 = 1.0  # Length of the first pendulum (m)
L2 = .4  # Length of the second pendulum (m)
m1 = 1.0  # Mass of the first pendulum (kg)
m2 = 40  # Mass of the second pendulum (kg)
g = 9.81  # Acceleration due to gravity (m/s^2)
b1 = 0.0 # Damping coefficient for the first pendulum
b2 = 0.0  # Damping coefficient for the second pendulum
initial_state = [np.pi/2, 0, np.pi/2, 0]  # Initial conditions [theta1, omega1, theta2, omega2]
t_span = (0, 10)  # Time span for simulation

# Solve the differential equations
solution = solve_ivp(double_pendulum, t_span, initial_state, args=(L1, L2, m1, m2, g, b1, b2), t_eval=np.linspace(0, 10, 1000))

# Plot the double pendulum motion
plot_double_pendulum(solution.t, solution.y[0], solution.y[2])

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
line1, = ax.plot([], [], 'o-', lw=2, markersize=6)
line2, = ax.plot([], [], 'o-', lw=2, markersize=6)

ani = FuncAnimation(fig, update, frames=len(solution.t), fargs=(line1, line2, solution.y[0], solution.y[2], L1, L2),
                    blit=True, interval=10)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Double Pendulum Motion Animation')
plt.show()
