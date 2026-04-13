Multi-Agent Prescribed Performance Control (PPC): Olympic Rings Formation
This repository contains a Python implementation of a multi-agent system where 5 robotic arms (2-DOF Planar Arms) collaborate to form the Olympic Games logo.

The trajectory control is achieved using Kinematic Prescribed Performance Control (PPC), which guarantees that the position tracking error of each robot's end-effector remains strictly within predefined, user-specified bounds throughout the entire movement.

1. Network Topology (Leader-Follower)
The system consists of 5 agents (1 Leader + 4 Followers). The Leader traces the central circular trajectory, while the Followers track the trajectory of their respective preceding agent, adding a constant spatial offset  Δij . This ensures the circles overlap correctly to form the Olympic logo:

Agent 0 (Leader): Center (Black Circle)
Agent 1 (Follower): Follows Agent 0 with offset  Δ10  (Blue Circle)
Agent 2 (Follower): Follows Agent 0 with offset  Δ20  (Red Circle)
Agent 3 (Follower): Follows Agent 1 with offset  Δ31  (Yellow Circle)
Agent 4 (Follower): Follows Agent 2 with offset  Δ42  (Green Circle)
The reference point for any follower  i  tracking agent  j  is given by:
pref,i=pj+Δij 

2. Robotic Arm Kinematics (2-DOF Planar Arm)
Each agent is modeled as a planar robotic arm with two links of lengths  l1  and  l2 . The control variables (states) are the joint angles:
q=[θ1,θ2]T 

The position of the end-effector in the Cartesian plane  p(q)=[x,y]T  is given by the Forward Kinematics:
p(q)=[l1cos(θ1)+l2cos(θ1+θ2)l1sin(θ1)+l2sin(θ1+θ2)] 

The relationship between the joint velocities  q˙  and the end-effector velocity  p˙  is described by the Jacobian Matrix  J(q) :
p˙=J(q)q˙ 

3. PPC Control Law (Prescribed Performance Control)
The controller's objective is to drive the end-effector of each arm to the desired trajectory  pref(t) , while maintaining the error  e(t)  strictly within a defined "performance funnel."

Step 1: Tracking Error
The position error for agent  i  is defined as:
e(t)=p(t)−pref(t) 

Step 2: Performance Function
We define a strictly decaying, positive function  ρ(t) , which establishes the error bounds:
ρ(t)=(ρ0−ρ∞)e−lt+ρ∞ 
Where:

ρ0 : Initial allowable error bound (must satisfy  |e(0)|<ρ0 ).
ρ∞ : Maximum allowable steady-state error (precision).
l : Exponential decay rate (convergence speed).
Step 3: Error Normalization & Transformation
The error is normalized as  ξ(t)=e(t)/ρ(t) . To ensure the error never hits the boundary ( |ξ|<1 ), we use a logarithmic transformation to map the constrained error to an unconstrained space  ϵ :
ϵ=12ln(1+ξ(t)1−ξ(t)) 

Step 4: Control Signal ( u )
The joint velocities are calculated using the Moore-Penrose pseudo-inverse ( J+ ) of the Jacobian:
u=q˙=−kJ+(q)ϵ 
Where  k  is a positive control gain.

4. Implementation Details
Language: Python 3.x
Dependencies: numpy, matplotlib
Integration: The script uses Euler integration to update joint states.
Safety: Includes a clamping mechanism on the normalized error to prevent numerical singularities at the performance boundaries.
5. Visualizing the Result
Upon execution, the script generates a plot showing the five trajectories. Each trajectory begins at a marked point ( ∗ ) and uses directional arrowheads to indicate the movement direction as the agents converge toward their respective formation offsets.
