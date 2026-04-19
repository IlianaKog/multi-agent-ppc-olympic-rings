import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import itertools

# --- 1. Inverse Kinematics Helper ---
def calculate_q0_from_xyz(target, base, l1=0.7, l2=0.7):
    """Calculates initial joint states q=[theta1, theta2, d3] for a target point."""
    dx, dy, dz = target[0] - base[0], target[1] - base[1], target[2]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(dist_sq)
    
    if dist >= (l1 + l2):
        q1, q2 = np.arctan2(dy, dx), 0.0
    else:
        cos_q2 = (dist_sq - l1**2 - l2**2) / (2 * l1 * l2)
        q2 = np.arccos(np.clip(cos_q2, -1.0, 1.0))
        q1 = np.arctan2(dy, dx) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return [q1, q2, dz]

# --- 2. 3D Robotic Arm Model ---
class PlanarArm3D:
    def __init__(self, base=[0.0, 0.0], l1=0.7, l2=0.7, q0=None):
        self.base = np.array(base)
        self.l1, self.l2 = l1, l2
        self.q = np.array(q0 if q0 is not None else [0.0, 0.0, 0.2], dtype=float)

    def get_pos(self):
        """End-effector Cartesian position."""
        x = self.l1 * np.cos(self.q[0]) + self.l2 * np.cos(self.q[0] + self.q[1])
        y = self.l1 * np.sin(self.q[0]) + self.l2 * np.sin(self.q[0] + self.q[1])
        return np.array([self.base[0] + x, self.base[1] + y, self.q[2]])

    def get_elbow_pos(self):
        """Elbow joint Cartesian position."""
        x, y = self.l1 * np.cos(self.q[0]), self.l1 * np.sin(self.q[0])
        return np.array([self.base[0] + x, self.base[1] + y, self.q[2]])

    def get_jacobian(self):
        """3x3 Analytical Jacobian."""
        s1, c1 = np.sin(self.q[0]), np.cos(self.q[0])
        s12, c12 = np.sin(self.q[0] + self.q[1]), np.cos(self.q[0] + self.q[1])
        return np.array([
            [-self.l1*s1 - self.l2*s12, -self.l2*s12, 0],
            [ self.l1*c1 + self.l2*c12,  self.l2*c12, 0],
            [ 0, 0, 1]
        ])

# --- 3. Prescribed Performance Controller (PPC) ---
class PPC3D:
    def __init__(self, arm, rho0, rho_inf, l_decay, k_gain):
        self.arm = arm
        self.rho0, self.rho_inf = np.array(rho0), np.array(rho_inf)
        self.l, self.k = l_decay, k_gain
        self.max_u = 5.0

    def control_step(self, p_ref, t, dt):
        p = self.arm.get_pos()
        e = p - p_ref
        rho = (self.rho0 - self.rho_inf) * np.exp(-self.l * t) + self.rho_inf
        xi = np.clip(e / rho, -0.99, 0.99)
        epsilon = 0.5 * np.log((1 + xi) / (1 - xi))
        J = self.arm.get_jacobian()
        u = -self.k * np.linalg.pinv(J) @ epsilon
        u_norm = np.linalg.norm(u)
        if u_norm > self.max_u: u = (u / u_norm) * self.max_u
        self.arm.q += u * dt
        return p

# --- 4. Simulation Configuration ---
dt, steps = 0.01, 1500
t_array = np.arange(0, steps * dt, dt)
R, omega, d, h = 1.0, -2.0, 2.2, 1.0

centers = {0: [0,0], 1: [-d,0], 2: [d,0], 3: [-d/2,-h], 4: [d/2,-h]}
init_targets = {0: [1.0,1.2,0.2], 1: [-1.0,1.4,0.2], 2: [3.0,1.4,0.2], 3: [0.0,2.0,0.2], 4: [2.0,-2.0,0.2]}
offsets = {"10": [-d,0,0], "20": [d,0,0], "31": [d/2,-h,0], "42": [-d/2,-h,0]}

agents = [PPC3D(PlanarArm3D(base=centers[i], q0=calculate_q0_from_xyz(init_targets[i], centers[i])), [5,5,5], [0.05]*3, 1.0, 5.0) for i in range(5)]
history, history_ref = [[] for _ in range(5)], [[] for _ in range(5)]
initial_tips = [a.arm.get_pos() for a in agents]

# Collision detection logging
agent_pairs = list(itertools.combinations(range(5), 2))
dist_tt, dist_ee, dist_te = {p: [] for p in agent_pairs}, {p: [] for p in agent_pairs}, {p: [] for p in agent_pairs}

# --- 5. Execution Loop (Cascaded Leader-Follower) ---
for t in t_array:
    alpha = 0.5 * (1 - np.cos(np.pi * (t / 2.0))) if t < 2.0 else 1.0
    z_ref = 0.2 if t < 2.0 else 0.0

    # Dynamic References based on Topology
    ref0_ideal = np.array([centers[0][0] + R*np.cos(omega*max(0, t-2)), centers[0][1] + R*np.sin(omega*max(0, t-2)), z_ref])
    pref0 = (1-alpha)*initial_tips[0] + alpha*ref0_ideal
    p0 = agents[0].control_step(pref0, t, dt)

    # Followers track preceding agent positions with offsets
    p1 = agents[1].control_step(np.array([*((1-alpha)*initial_tips[1][:2] + alpha*(p0[:2] + offsets["10"][:2])), z_ref]), t, dt)
    p2 = agents[2].control_step(np.array([*((1-alpha)*initial_tips[2][:2] + alpha*(p0[:2] + offsets["20"][:2])), z_ref]), t, dt)
    p3 = agents[3].control_step(np.array([*((1-alpha)*initial_tips[3][:2] + alpha*(p1[:2] + offsets["31"][:2])), z_ref]), t, dt)
    p4 = agents[4].control_step(np.array([*((1-alpha)*initial_tips[4][:2] + alpha*(p2[:2] + offsets["42"][:2])), z_ref]), t, dt)

    tips = [p0, p1, p2, p3, p4]
    elbows = [a.arm.get_elbow_pos() for a in agents]
    for i in range(5):
        history[i].append(tips[i])
        # Reconstruct tracking references for error analysis
        if i == 0: history_ref[i].append(pref0)
        elif i == 1: history_ref[i].append(np.array([*(p0[:2]+offsets["10"][:2]), z_ref]))
        elif i == 2: history_ref[i].append(np.array([*(p0[:2]+offsets["20"][:2]), z_ref]))
        elif i == 3: history_ref[i].append(np.array([*(p1[:2]+offsets["31"][:2]), z_ref]))
        elif i == 4: history_ref[i].append(np.array([*(p2[:2]+offsets["42"][:2]), z_ref]))

    for i, j in agent_pairs:
        dist_tt[(i,j)].append(np.linalg.norm(tips[i] - tips[j]))
        dist_ee[(i,j)].append(np.linalg.norm(elbows[i] - elbows[j]))
        dist_te[(i,j)].append(min(np.linalg.norm(tips[i]-elbows[j]), np.linalg.norm(elbows[i]-tips[j])))

history, history_ref = np.array(history), np.array(history_ref)
colors = ['black', 'blue', 'red', 'gold', 'green']

# --- 6. Plotting Results ---

# FIGURE 1: 3D Workspace Visualization
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
for i in range(5):
    h = history[i]
    contact_idx = np.where(h[:, 2] < 0.01)[0]
    split = contact_idx[0] if len(contact_idx) > 0 else len(h)
    ax1.plot(h[:split, 0], h[:split, 1], h[:split, 2], color=colors[i], ls='--', lw=1.2)
    ax1.plot(h[split:, 0], h[split:, 1], h[split:, 2], color=colors[i], ls='-', lw=2.2, label=f'Agent {i}')
ax1.set_title("3D Trajectories: Cascaded Leader-Follower PPC"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.legend(); plt.grid(True)

# FIGURE 2: 2D Top-Down View
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(history[i, :, 0], history[i, :, 1], color=colors[i], lw=2, label=f'Agent {i}')
    plt.plot(centers[i][0], centers[i][1], 'ks', markersize=4, alpha=0.3)
plt.title("2D Projection: Olympic Ring Formation"); plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.axis('equal'); plt.grid(True); plt.legend(loc='upper right')

# FIGURE 3: PPC Error Envelopes (Detailed Analysis)
fig3, axes3 = plt.subplots(5, 3, figsize=(15, 18), sharex=True)
dims = ['X', 'Y', 'Z']
for i in range(5):
    rho_t = (agents[i].rho0[0] - agents[i].rho_inf[0]) * np.exp(-agents[i].l * t_array) + agents[i].rho_inf[0]
    for d_idx in range(3):
        ax = axes3[i, d_idx]
        error = history[i, :, d_idx] - history_ref[i, :, d_idx]
        ax.plot(t_array, error, color=colors[i], label='Error')
        ax.plot(t_array, rho_t, 'r--', alpha=0.5, label='Bounds')
        ax.plot(t_array, -rho_t, 'r--', alpha=0.5)
        if i == 0: ax.set_title(f"Dimension {dims[d_idx]}")
        if d_idx == 0: ax.set_ylabel(f"Agent {i}\nError (m)")
        ax.grid(True, linestyle=':')
axes3[-1, 1].set_xlabel("Time (s)")
plt.tight_layout()

# FIGURE 4: Collision Monitoring (Safety Diagnostics)
fig4, axes4 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
diag_titles = ["Tip-to-Tip Distance", "Elbow-to-Elbow Distance", "Tip-to-Elbow Distance"]
for ax, data, title in zip(axes4, [dist_tt, dist_ee, dist_te], diag_titles):
    for pair in agent_pairs: ax.plot(t_array, data[pair], alpha=0.6, label=f"P{pair}")
    ax.axhline(0.15, color='red', ls='--', lw=2, label="Safety Limit")
    ax.set_title(title); ax.set_ylabel("Dist (m)"); ax.grid(True)
axes4[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()
