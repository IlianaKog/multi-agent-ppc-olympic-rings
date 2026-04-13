# Multi-Agent Prescribed Performance Control (PPC): Olympic Rings Formation

This repository contains a Python implementation of a multi-agent system where 5 robotic arms (2-DOF Planar Arms) collaborate to form the Olympic Games logo. 

The trajectory control is achieved using **Kinematic Prescribed Performance Control (PPC)**, which guarantees that the position tracking error of each robot's end-effector remains strictly within predefined, user-specified bounds throughout the entire movement.

## 1. Network Topology (Leader-Follower)
The system consists of 5 agents (1 Leader + 4 Followers). The Leader traces the central circular trajectory, while the Followers track the trajectory of their respective preceding agent, adding a constant spatial offset $\Delta_{ij}$. This ensures the circles overlap correctly to form the Olympic logo:

* **Agent 0 (Leader):** Center (Black Circle)
* **Agent 1 (Follower):** Follows Agent 0 with offset $\Delta_{10}$ (Blue Circle)
* **Agent 2 (Follower):** Follows Agent 0 with offset $\Delta_{20}$ (Red Circle)
* **Agent 3 (Follower):** Follows Agent 1 with offset $\Delta_{31}$ (Yellow Circle)
* **Agent 4 (Follower):** Follows Agent 2 with offset $\Delta_{42}$ (Green Circle)

The reference point for any follower $i$ tracking agent $j$ is given by:
$$p_{ref, i} = p_j + \Delta_{ij}$$

## 2. Robotic Arm Kinematics (2-DOF Planar Arm)
Each agent is modeled as a planar robotic arm with two links of lengths $l_1$ and $l_2$. The control variables (states) are the joint angles:
$$q = [\theta_1, \theta_2]^T$$

The position of the end-effector in the Cartesian plane $p(q) = [x, y]^T$ is given by the **Forward Kinematics**:
$$p(q) = \begin{bmatrix} l_1 \cos(\theta_1) + l_2 \cos(\theta_1+\theta_2) \\ l_1 \sin(\theta_1) + l_2 \sin(\theta_1+\theta_2) \end{bmatrix}$$

The relationship between the joint velocities $\dot{q}$ and the end-effector velocity $\dot{p}$ is described by the **Jacobian Matrix** $J(q)$:
$$\dot{p} = J(q) \dot{q}$$

## 3. PPC Control Law (Prescribed Performance Control)
The controller's objective is to drive the end-effector of each arm to the desired trajectory $p_{ref}(t)$, while maintaining the error $e(t)$ strictly within a defined "performance funnel."

### Step 1: Tracking Error
The position error for agent $i$ is defined as:
$$e(t) = p(t) - p_{ref}(t)$$

### Step 2: Performance Function
We define a strictly decaying, positive function $\rho(t)$, which establishes the error bounds:
$$\rho(t) = (\rho_0 - \rho_\infty)e^{-lt} + \rho_\infty$$
Where:
* **$\rho_0$**: Initial allowable error bound (must satisfy $|e(0)| < \rho_0$).
* **$\rho_\infty$**: Maximum allowable steady-state error (precision).
* **$l$**: Exponential decay rate (convergence speed).

### Step 3: Error Normalization & Transformation
The error is normalized as $\xi(t) = e(t)/\rho(t)$. To ensure the error never hits the boundary ($|\xi| < 1$), we use a logarithmic transformation to map the constrained error to an unconstrained space $\epsilon$:
$$\epsilon = \frac{1}{2} \ln \left( \frac{1 + \xi(t)}{1 - \xi(t)} \right)$$

### Step 4: Control Signal ($u$)
The joint velocities are calculated using the Moore-Penrose pseudo-inverse ($J^+$) of the Jacobian:
$$u = \dot{q} = -k J^+(q) \epsilon$$
Where $k$ is a positive control gain.

## 4. Implementation Details
* **Language:** Python 3.x
* **Dependencies:** `numpy`, `matplotlib`
* **Integration:** The script uses Euler integration to update joint states.
* **Safety:** Includes a clamping mechanism on the normalized error to prevent numerical singularities at the performance boundaries.

## 5. Visualizing the Result
Upon execution, the script generates a plot showing the five trajectories. Each trajectory begins at a marked point ($*$) and uses directional arrowheads to indicate the movement direction as the agents converge toward their respective formation offsets.
