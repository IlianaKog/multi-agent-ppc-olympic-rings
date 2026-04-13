import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class PlanarArm:
    def __init__(self, l1=2.0, l2=2.0, q0=None):
        self.l1 = l1
        self.l2 = l2
        self.q = np.array(q0 if q0 is not None else [np.pi/4, -np.pi/2], dtype=float)

    def get_pos(self):
        """Returns the (x,y) position of the end-effector."""
        x = self.l1 * np.cos(self.q[0]) + self.l2 * np.cos(self.q[0] + self.q[1])
        y = self.l1 * np.sin(self.q[0]) + self.l2 * np.sin(self.q[0] + self.q[1])
        return np.array([x, y])

    def get_jacobian(self):
        """Calculates the 2x2 Jacobian matrix."""
        j11 = -self.l1 * np.sin(self.q[0]) - self.l2 * np.sin(self.q[0] + self.q[1])
        j12 = -self.l2 * np.sin(self.q[0] + self.q[1])
        j21 = self.l1 * np.cos(self.q[0]) + self.l2 * np.cos(self.q[0] + self.q[1])
        j22 = self.l2 * np.cos(self.q[0] + self.q[1])
        return np.array([[j11, j12], [j21, j22]])

class PPCAgent:
    def __init__(self, arm, rho0, rho_inf, l_decay, k_gain):
        self.arm = arm
        self.rho0 = np.array(rho0)
        self.rho_inf = np.array(rho_inf)
        self.l = l_decay
        self.k = k_gain

    def control_step(self, p_ref, t, dt):
        p = self.arm.get_pos()
        e = p - p_ref

        # Performance Function bounds
        rho = (self.rho0 - self.rho_inf) * np.exp(-self.l * t) + self.rho_inf

        # Normalized Error
        xi = e / rho
        # Clamping to avoid mathematical domain errors at the boundaries
        xi = np.clip(xi, -0.99, 0.99) 
        
        # Error Transformation (Inverse Hyperbolic Tangent / Logarithmic)
        epsilon = 0.5 * np.log((1 + xi) / (1 - xi))

        # Kinematic Control Law
        J = self.arm.get_jacobian()
        J_pinv = np.linalg.pinv(J) # Moore-Penrose pseudo-inverse
        
        # Calculate desired joint velocities (u = q_dot)
        u = -self.k * J_pinv @ epsilon
        
        # Update arm states (Euler integration)
        self.arm.q += u * dt
        return p

# --- Simulation Setup ---
dt = 0.01
time_steps = 1000
t_array = np.arange(0, time_steps * dt, dt)

# Olympic Rings Parameters
R = 1.0     # Circle radius
omega = 2.0 # Circle tracking frequency
d = 2.2     # Horizontal distance between top circles
h = 1.0     # Vertical offset for bottom circles

# Create the 5 arms - starting from different initial angles for realistic transient response
agents = [
    PPCAgent(PlanarArm(q0=[1.0, 1.0]), rho0=[5,5], rho_inf=[0.05, 0.05], l_decay=1.0, k_gain=5.0), # 0: Leader
    PPCAgent(PlanarArm(q0=[1.5, 0.5]), rho0=[5,5], rho_inf=[0.05, 0.05], l_decay=1.0, k_gain=5.0), # 1: Follows 0
    PPCAgent(PlanarArm(q0=[0.5, 1.5]), rho0=[5,5], rho_inf=[0.05, 0.05], l_decay=1.0, k_gain=5.0), # 2: Follows 0
    PPCAgent(PlanarArm(q0=[1.0, 0.5]), rho0=[5,5], rho_inf=[0.05, 0.05], l_decay=1.0, k_gain=5.0), # 3: Follows 1
    PPCAgent(PlanarArm(q0=[0.0, 2.0]), rho0=[5,5], rho_inf=[0.05, 0.05], l_decay=1.0, k_gain=5.0)  # 4: Follows 2
]

# Offsets Δ_{ij}
delta_10 = np.array([-d, 0])
delta_20 = np.array([d, 0])
delta_31 = np.array([d/2, -h])
delta_42 = np.array([-d/2, -h])

history = [[] for _ in range(5)]

# --- Simulation Loop ---
for t in t_array:
    # 1. Leader (Agent 0) traces the central circle
    pref_0 = np.array([R * np.cos(omega * t), R * np.sin(omega * t)])
    p0 = agents[0].control_step(pref_0, t, dt)
    
    # 2. Followers (Agents 1 & 2) track Agent 0 with corresponding offsets
    p1 = agents[1].control_step(p0 + delta_10, t, dt)
    p2 = agents[2].control_step(p0 + delta_20, t, dt)
    
    # 3. Followers (Agents 3 & 4) track Agents 1 & 2 respectively
    p3 = agents[3].control_step(p1 + delta_31, t, dt)
    p4 = agents[4].control_step(p2 + delta_42, t, dt)
    
    # Save trajectory history for plotting
    history[0].append(p0)
    history[1].append(p1)
    history[2].append(p2)
    history[3].append(p3)
    history[4].append(p4)

# --- Plotting ---
history = np.array(history)
colors = ['black', 'blue', 'red', 'yellow', 'green']
arrow_interval = 200 # How sparse the arrows will be

plt.figure(figsize=(10, 6))
ax = plt.gca()

for i in range(5):
    # Plot the main trajectory line
    plt.plot(history[i, :, 0], history[i, :, 1], color=colors[i], linewidth=2, label=f'Agent {i}')
    
    # --- ADD * AT t=0 ---
    plt.plot(history[i, 0, 0], history[i, 0, 1], marker='*', color='black', markersize=5, linestyle='None')
    
    # Add the rest of the arrowheads sparsely along the path
    for j in range(arrow_interval, time_steps, arrow_interval):
        tail = history[i, j - 1] 
        head = history[i, j]     

        arrow = FancyArrowPatch(tail, head,
                                arrowstyle='->', # Triangle tip
                                mutation_scale=20, # Size of the arrowhead
                                edgecolor='black',
                                facecolor='none',
                                linewidth=0.8,       # 0 thickness removes the arrow's tail/line
                                antialiased=True)
        ax.add_patch(arrow)

plt.axis('equal')
plt.title("Multi-Agent PPC: Olympic Rings with Directional Arrowheads (Including t=0)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
