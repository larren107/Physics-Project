import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate_collision(m1, m2, v1_initial, v2_initial, collision_time, total_time, dt, restitution):
    # Initialize arrays to store data
    time = np.arange(0, total_time, dt)
    v1 = np.zeros_like(time)
    v2 = np.zeros_like(time)

    # Set initial velocities
    v1[0] = v1_initial
    v2[0] = v2_initial

    # Simulate the collision
    collision_index = int(collision_time / dt)
    for i in range(1, len(time)):
        if i == collision_index:
            # Calculate velocities after collision
            v1_final = ((m1 - m2 * restitution) * v1_initial + (1 + restitution) * m2 * v2_initial) / (m1 + m2)
            v2_final = (m1 * (1 + restitution) * v1_initial + (m2 - m1 * restitution) * v2_initial) / (m1 + m2)
            v1[i] = v1_final
            v2[i] = v2_final
        else:
            v1[i] = v1[i - 1]
            v2[i] = v2[i - 1]

    return time, v1, v2


# Function to update animation
def update(frame):
    # Update mass positions
    mass1.set_data(time[frame], v1[frame])
    mass2.set_data(time[frame], v2[frame])

    # Update velocity lines
    line1.set_data([time[frame], time[frame]], [v1_initial, v1[frame]])
    line2.set_data([time[frame], time[frame]], [v2_initial, v2[frame]])

    return mass1, mass2, line1, line2


# Define parameters
m1 = 100.0  # Mass of object 1 (kg)
m2 = 100.0  # Mass of object 2 (kg)
v1_initial = 3.0  # Initial velocity of object 1 (m/s)
v2_initial = -5.0  # Initial velocity of object 2 (m/s)
collision_time = 2.0  # Time of collision (s)
total_time = 5.0  # Total simulation time (s)
dt = 0.01  # Time step (s)
restitution = .4 # Coefficient of restitution (1 for elastic collision, 0 for completely inelastic)

# Simulate collision
time, v1, v2 = simulate_collision(m1, m2, v1_initial, v2_initial, collision_time, total_time, dt, restitution)

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(0, total_time)
ax.set_ylim(min(v1_initial, v2_initial) - 1, max(v1_initial, v2_initial) + 1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Collision Animation')

# Plot initial positions
mass1, = ax.plot(0, v1_initial, 'bo', markersize=10, label='Object 1')
mass2, = ax.plot(0, v2_initial, 'ro', markersize=10, label='Object 2')
line1, = ax.plot([0, 0], [v1_initial, v1_initial], 'b--')
line2, = ax.plot([0, 0], [v2_initial, v2_initial], 'r--')

ax.legend()

ani = FuncAnimation(fig, update, frames=len(time), interval=dt * 1000, blit=True)

plt.show()