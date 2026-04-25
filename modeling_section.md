# Modeling

## Environment and Robot Modelling

### Problem Formulation

The locomotion problem is formulated as a Markov decision process (MDP)
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma),$$
where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P$ is the transition model induced by the MuJoCo simulator, $r$ is the reward function, and $\gamma$ is the discount factor. Each episode consists of a sequence of states, actions, observations, and rewards, with episodes terminating upon robot fall or reaching maximum episode length of 1000 control steps (20 seconds of simulated time).

### Robot Morphology and Kinematics

The robot used in this work is the Unitree Go1 quadruped, modeled as a floating-base rigid-body system with 12 actuated joints and a 6-DoF free base. The total mass is $m_{\text{total}} = 5.204$ kg, distributed across trunk (mass 5.204 kg) and four identical legs. Each leg contains three revolute joints:

- **Abduction joint** $q_{\text{abd}} \in [-0.863, 0.863]$ rad (controls lateral leg motion)
- **Hip joint** $q_{\text{hip}} \in [-0.686, 4.501]$ rad (primary forward/backward leg actuation)  
- **Knee joint** $q_{\text{knee}} \in [-2.818, -0.888]$ rad (controls leg length and ground clearance)

The generalized configuration and velocity are
$$q = \begin{bmatrix} p_b \\ q_b \\ q_j \end{bmatrix} \in \mathbb{R}^{19},\qquad
\nu = \begin{bmatrix} v_b \\ \dot{q}_j \end{bmatrix} \in \mathbb{R}^{18},$$
where $p_b = [x, y, z]^\top \in \mathbb{R}^3$ is the base position, $q_b \in \mathbb{R}^4$ is the base quaternion (orientation), $q_j = [q_{FR}, q_{FL}, q_{RR}, q_{RL}]^\top \in \mathbb{R}^{12}$ is the joint position vector (three joints per leg), $v_b = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^\top \in \mathbb{R}^6$ is the base spatial velocity (linear plus angular), and $\dot{q}_j \in \mathbb{R}^{12}$ is the joint velocity vector. The legs are arranged as: FR (front-right), FL (front-left), RR (rear-right), RL (rear-left).

The robot dynamics follow the standard floating-base rigid-body equation
$$M(q)\dot{\nu} + h(q,\nu) = S^\top \tau + J_c(q)^\top \lambda,$$
where $M(q) \in \mathbb{R}^{18 \times 18}$ is the mass matrix, $h(q,\nu) \in \mathbb{R}^{18}$ contains Coriolis, centrifugal, and gravitational terms, $S \in \mathbb{R}^{18 \times 12}$ is the actuator selection matrix (identity block for joints, zero for base), $\tau \in \mathbb{R}^{12}$ is the actuator torque vector, $J_c(q) \in \mathbb{R}^{3n_c \times 18}$ is the contact Jacobian where $n_c$ is the number of active contacts, and $\lambda \in \mathbb{R}^{3n_c}$ is the ground contact force vector. Contact forces are constrained to satisfy unilateral contact (forces can only push), friction cones, and non-penetration constraints. This model is not derived manually; it is adopted from the MuJoCo rigid-body physics engine with Euler integration. The inertial properties are specified in the MJCF model file and include limb masses: trunk 5.204 kg, hip link 0.68 kg, thigh link 1.009 kg, calf link 0.196 kg per leg.

### Simulation Environment and Contact Model

The task is executed on a flat terrain modeled as a plane with friction coefficient $\mu = 0.6$. The contact model uses MuJoCo's elliptic cone friction with standard Coulomb friction law. In the selected "feet-only" environment variant, only the four foot spheres (radius 0.023 m) are allowed to make contact with the ground via collision geometry coupling. All other body segments have collision detection disabled globally, which simplifies contact handling and prevents undesired self-collisions or body-ground penetrations from corrupting learning.

The simulation step is
$$\Delta t_{\text{sim}} = 0.004 \text{ s},$$
with 1 Newton-Euler step and 5 linear solver iterations per step. The control step is
$$\Delta t = 0.02 \text{ s},$$
so each control action is held for 5 physics substeps. Each episode lasts 1000 control steps, corresponding to 20 seconds of simulated time. Initial positions are randomized: $\Delta x, \Delta y \sim \mathcal{U}(-0.5, 0.5)$ m and $\Delta \text{yaw} \sim \mathcal{U}(-\pi, \pi)$; initial velocities are $\in \mathcal{U}(-0.5, 0.5)$ m/s or rad/s.

### Sensor Suite

The robot is equipped with an inertial measurement unit (IMU) located at the torso with the following sensors (sampling rate: 250 Hz, fused into state at control rate):

- **Accelerometer**: Raw linear acceleration in IMU frame, drift-free
- **Gyroscope (angular velocity sensor)**: Angular velocity vector $\omega = [\omega_x, \omega_y, \omega_z]^\top$ in IMU frame
- **Inclinometer** (derived from body orientation): Gravity vector measured in body frame $g^b = [g_x^b, g_y^b, g_z^b]^\top$ (normalized to unit magnitude if gravity acceleration is $g = 9.81$ m/s$^2$)

Additionally, foot-contact sensors on each foot detect binary ground contact via MuJoCo's contact force sensing, and foot kinematics (positions in world frame relative to IMU) are computed from forward kinematics.

### Actuation and Control Architecture  

The action vector is
$$a_t \in [-1,1]^{12},$$
with one action for each joint actuator (three actions per leg in order FR, FL, RR, RL). The policy does not command torque directly; instead, it outputs a joint position offset around a nominal standing home pose:
$$q_{j,t}^{\text{ref}} = q_j^{\text{home}} + s_a a_t,\qquad s_a = 0.5 \text{ rad}.$$
The nominal home posture (from MuJoCo keyframe) is $q_j^{\text{home}} = [0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8]^\top$ rad, which corresponds to a standing posture with slightly bent knees.

The low-level actuator is a proportional-derivative position controller implemented at the actuator level:
$$\tau_t = K_p\left(q_{j,t}^{\text{ref}} - q_{j,t}\right) - K_d \dot{q}_{j,t},$$
where $K_p = 35$ N·m/rad and $K_d = 0.5$ N·m·s/rad are the proportional and derivative gains. The force range is clipped to $[-23.7, 23.7]$ N·m per joint (knee joints allow $[-35.55, 35.55]$ N·m). Thus the control pipeline is: policy $a_t$ → position target $q_{\text{ref}}$ → PD controller → clipped torque $\tau_t$ → robot dynamics.

### Locomotion Command and Task Definition

The locomotion command at time $t$ is
$$c_t = \begin{bmatrix} v_{x,t}^{\text{cmd}} & v_{y,t}^{\text{cmd}} & \omega_{z,t}^{\text{cmd}} \end{bmatrix}^\top,$$
where $v_x^{\text{cmd}}$ and $v_y^{\text{cmd}}$ are commanded body-frame linear velocities (m/s) and $\omega_z^{\text{cmd}}$ is the commanded yaw rate (rad/s). These commands are externally supplied (via joystick or teleoperation in deployment, or randomly sampled during training). Commands are resampled stochastically during an episode according to an exponentially distributed inter-arrival time. The command update law used in the environment is
$$c_{k+1} = c_k - w_k \odot \left(c_k - y_k \odot z_k\right),$$
with
$$y_k \sim \mathcal{U}(-a,a), \qquad
z_k \sim \text{Bernoulli}(b), \qquad
w_k \sim \text{Bernoulli}(0.5),$$
where $\odot$ denotes elementwise multiplication, $a = [1.5, 0.8, 1.2]^\top$ m/s and rad/s, and $b = [0.9, 0.25, 0.5]^\top$. This gives a randomized command generator where forward velocity has high persistence (90%), lateral velocity resets more often (25%), and yaw resets with 50% probability.

### Observation Space and Sensor Noise

#### Reference Frames and Coordinates

Throughout the model, three reference frames are used:
- **World frame** ($W$): Inertial frame fixed to the ground plane, with z-axis pointing up
- **Body frame** ($B$): Non-inertial frame attached to the torso (trunk) center of mass, with x-axis pointing forward, y-axis pointing left, z-axis pointing up
- **IMU frame** ($I$): Attached to the IMU sensor location (offset from CoM), typically coincident with body frame for pose estimation

Sensor measurements from the IMU are expressed in body frame. The transformation from body to world frame is given by the rotation matrix $R_{WB}(q_b)$ derived from the base quaternion. Linear velocities are transformed as $v^W = R_{WB} v^B$, and similarly for angular velocities and forces.

The actor observation is a 48-dimensional vector
$$o_t =
\begin{bmatrix}
\tilde{v}_t^b \\
\tilde{\omega}_t^b \\
\tilde{g}_t^b \\
\tilde{q}_{j,t} - q_j^{\text{home}} \\
\widetilde{\dot{q}}_{j,t} \\
a_{t-1} \\
c_t
\end{bmatrix}
\in \mathbb{R}^{48},$$
with component breakdown:
- $\tilde{v}_t^b \in \mathbb{R}^3$ (3 dims): noisy body-frame linear velocity from IMU  
- $\tilde{\omega}_t^b \in \mathbb{R}^3$ (3 dims): noisy gyroscope angular velocity
- $\tilde{g}_t^b \in \mathbb{R}^3$ (3 dims): noisy gravity vector in body frame (inclinometer)
- $\tilde{q}_{j,t} - q_j^{\text{home}} \in \mathbb{R}^{12}$ (12 dims): noisy joint positions relative to home pose
- $\widetilde{\dot{q}}_{j,t} \in \mathbb{R}^{12}$ (12 dims): noisy joint velocities
- $a_{t-1} \in \mathbb{R}^{12}$ (12 dims): previous action (for temporal context)
- $c_t \in \mathbb{R}^{3}$ (3 dims): current locomotion command

Uniform sensor noise is injected during training via
$$\tilde{x}_t = x_t + \sigma_{\text{level}} \cdot n_{\text{scale}} \cdot u_t,$$
where $x_t$ is the true measurement, $\sigma_{\text{level}} = 1.0$ is the noise level multiplier, $n_{\text{scale}}$ is the modality-specific scale factor, and $u_t \sim \mathcal{U}(-1, 1)$ is uniform random noise. Modality-specific noise scales are:

| Sensor | Scale $n_{\text{scale}}$ |
|--------|----------|
| Joint position | 0.03 rad |
| Joint velocity | 1.5 rad/s |
| Gyroscope | 0.2 rad/s |
| Inclinometer (gravity) | 0.05 (normalized magnitude) |
| Linear velocity | 0.1 m/s |

This noise helps the learned policy generalize to real-world sensor measurements and reduces overfitting to noise-free simulation.

An asymmetric actor-critic structure is employed. The policy $\pi_\theta$ receives only the 48-dimensional $o_t$ and operates with onboard-realizable observations (compatible with real robot deployment). The critic $V_\phi$ receives a privileged 123-dimensional state vector
$$o_t^v \in \mathbb{R}^{123},$$
which includes the full $o_t$ plus additional noise-free, simulation-only privileged information:

- Clean gyro, accelerometer, gravity (no noise): +9 dims
- Clean position, velocity relative to home: +24 dims  
- Global (world frame) linear velocity: +3 dims
- Global angular velocity: +3 dims
- Actuator forces (torques applied by controllers): +12 dims
- Foot contact states (4 feet): +4 dims
- Foot linear velocities (xy components): +8 dims
- Foot air time (accumulated swing duration): +4 dims
- External perturbation force on torso (if enabled): +3 dims
- Perturbation indicator (boolean): +1 dim

This privileged state improves value function learning, leading to better advantage estimates and more stable training, while the policy remains deployable from onboard sensors only (learning with domain gap).

### Episode Termination  

The episode terminates when the robot falls, modeled as
$$d_t =
\begin{cases}
1, & \text{if } u_{z,t} < 0, \\
0, & \text{otherwise},
\end{cases}$$
where $u_t = [u_{x,t},u_{y,t},u_{z,t}]^\top$ is the torso (trunk body) up-vector (z-axis) in the world frame, obtained from the body orientation quaternion. A negative z-component indicates the robot is tilted more than 90 degrees and is considered fallen.

## PPO Model and Reward Function

### Policy and Value Function Architecture

The control policy is learned using Proximal Policy Optimization (PPO). The actor is a stochastic policy represented by a neural network:
$$\pi_\theta(a_t \mid o_t) = \mathcal{N}(\mu_\theta(o_t), \sigma^2(o_t)),$$
where $\mu_\theta: \mathbb{R}^{48} \to \mathbb{R}^{12}$ outputs mean action and $\sigma: \mathbb{R}^{48} \to \mathbb{R}^{12}$ outputs per-dimension standard deviations (log-variance actually stored and exponentiated).

#### Policy Network (Actor)

The policy network is a multilayer perceptron (MLP) with:
- **Depth (number of layers)**: 3 hidden layers (4 total including input-to-first, output from last)
- **Width configuration** (hidden layer sizes): $(512, 256, 128)$ neurons per layer
- **Total parameters**: Approximately $[48 \times 512 + 512 \times 256 + 256 \times 128 + 128 \times 12] + \text{biases} \approx 184K$ parameters
- **Activation functions**: 
  - Hidden layers: Tanh (smooth, bounded, avoids ReLU dying problem)
  - Output mean: Tanh followed by affine scaling to $[-1, 1]$ action range
  - Output log-variance: Separate head (linear), clamped to $[\log(0.01), \log(0.5)]$ to bound exploration
- **Weight initialization**: Orthogonal initialization for policy/value networks (Brax default), ensures stable gradient flow
- **Output parameterization**: 
  - Mean $\mu_\theta(o_t) = \tanh(h_3) \in [-1, 1]^{12}$ where $h_3$ is final hidden activation
  - Std dev $\sigma_i = \exp(\text{clamp}(\sigma'_i, -4.61, -0.69))$ to keep $\sigma \in [0.01, 0.5]$

#### Value Function Network (Critic)

The value function is represented as a separate MLP:
$$V_\phi(o_t^v): \mathbb{R}^{123} \to \mathbb{R},$$
with architecture:
- **Depth (number of layers)**: 3 hidden layers (4 total including input-to-first)
- **Width configuration** (hidden layer sizes): $(512, 256, 128)$ neurons per layer
- **Total parameters**: Approximately $[123 \times 512 + 512 \times 256 + 256 \times 128 + 128 \times 1] + \text{biases} \approx 216K$ parameters
- **Activation functions**:
  - Hidden layers: Tanh (matches policy network for consistency)
  - Output layer: Linear (unbounded scalar value estimate)
- **Weight initialization**: Orthogonal initialization (same as policy)
- **Special capabilities**: Access to asymmetric privileged observations ($o_t^v$) during training only; not available at deployment

#### Joint Training

Both policy and value networks are trained jointly with shared computational graph but separate parameter sets. Key design details:
- **Separate networks**: Policy and critic do not share hidden layers (unlike single-head architectures), allowing specialized feature learning
- **Gradient flow**: Both networks receive gradients from their respective loss terms simultaneously
- **Weight sharing**: None (fully independent networks) — this asymmetry (policy sees $o_t$, critic sees $o_t^v$) is deliberate
- **Initialization**: Both use orthogonal initialization with gain adjusted per layer to stabilize deep networks (typically gain ~1.4 for Tanh)

### Training Procedure and Hyperparameters

#### PPO Objective and Loss Formulation

The PPO objective is the clipped surrogate objective
$$L^{\text{PPO}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\varrho_t(\theta)\hat{A}_t,\;
\text{clip}\big(\varrho_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat{A}_t
\right)
\right],$$
where the probability ratio is
$$\varrho_t(\theta)
=
\frac{\pi_\theta(a_t \mid o_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid o_t)},$$
$\hat{A}_t$ is the estimated generalized advantage (GAE), and $\epsilon$ is the clipping parameter. This clipped objective prevents excessively large policy updates that could destabilize learning.

Advantages are computed using generalized advantage estimation (GAE) with
$$\hat{A}_t = \sum_{\ell=0}^{\infty}(\gamma\lambda)^\ell \delta_t^\ell, \qquad \delta_t^{\ell} = r_{t+\ell} + \gamma V_\phi(o_{t+\ell+1}^v) - V_\phi(o_{t+\ell}^v),$$
where $\lambda = 0.95$ is the GAE decay parameter (bias-variance tradeoff). In practice, advantages are truncated at the episode boundary and normalized (mean-centered and divided by std dev + $\epsilon$) for stability.

The value loss is computed as
$$L^{V}(\phi) = \mathbb{E}_t \left[\left(V_\phi(o_t^v)-\hat{R}_t\right)^2\right],$$
where $\hat{R}_t$ is the Monte-Carlo target return computed via discounted cumulative rewards:
$$\hat{R}_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}.$$

The entropy regularization term is
$$H(\pi_\theta) = \mathbb{E}_{o_t}\left[-\sum_i \sigma_i \log \sigma_i\right],$$
for a Gaussian policy, encouraging exploration by penalizing deterministic policies.

The overall training loss combined is
$$L(\theta,\phi)
=
- L^{\text{PPO}}(\theta)
+ c_v \, L^{V}(\phi)
- c_e \, H(\pi_\theta),$$
where $c_v$ is the value-loss coefficient and $c_e$ is the entropy coefficient. Minimizing $-L^{\text{PPO}}$ maximizes the clipped objective.

#### Learning Rate and Optimization

**Learning Rate Configuration**:
- **Base learning rate**: $\eta = 3 \times 10^{-4}$ (0.0003)
- **Optimizer**: Adam with default parameters ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Learning rate schedule**: Fixed (no decay or annealing)
- **Gradient clipping**: Max L2 norm of 1.0 per network per update step
- **Numerics**: All computations in float32 (JAX default), with stable log-probability computation

The fixed learning rate of $3 \times 10^{-4}$ was tuned for the Go1 task and balances convergence speed with stability. Larger learning rates ($>10^{-3}$) can cause policy instability and entropy collapse; smaller rates ($<10^{-4}$) lead to slow convergence. Gradient clipping with norm 1.0 prevents catastrophic updates from large temporal-difference errors.

#### PPO Hyperparameter Summary

| Parameter | Value | Description |
|---|---|---|
| **Learning rate** ($\eta$) | $3 \times 10^{-4}$ | Adam optimizer step size; tuned for stable convergence |
| **Discount factor** ($\gamma$) | 0.97 | Information decay: 97% of future reward retained per step (short-horizon) |
| **GAE decay** ($\lambda$) | 0.95 | Advantage estimation tradeoff: low = high-bias/low-variance; high = low-bias/high-variance |
| **Clipping range** ($\epsilon$) | 0.2 | Policy ratio clipped to $[0.8, 1.2] \times \hat{A}_t$; prevents excessive updates |
| **Entropy coefficient** ($c_e$) | $0.01$ | Entropy bonus: $-0.01 \times H[\pi]$ added to loss; weak exploration regularization |
| **Value loss weight** ($c_v$) | 1.0 | Value TD loss weighted equally to policy loss (unity weighting) |
| **Max gradient norm** | 1.0 | L2 norm clipping for gradient per network; prevents divergence |
| **Adam $\beta_1$** | 0.9 | Exponential decay rate for 1st moment (velocity) |
| **Adam $\beta_2$** | 0.999 | Exponential decay rate for 2nd moment (acceleration) |
| **Adam $\epsilon$** | $10^{-8}$ | Numerical stability constant for adaptive learning rate |
| **Reward scaling** | 1.0 | No reward normalization (raw rewards used); clipped to [0, 10000] |
| **Action repeat** | 1 | No action repetition; new action computed every control step |

#### Training Data Collection and Optimization

**Parallelization and Batch Composition**:
- **Number of parallel environments**: 8192 (vectorized across batch dimension)
- **Rollout horizon** (unroll length): 20 control steps per trajectory segment (0.4 seconds)
- **Total transitions per gradient step**: $8192 \times 20 = 163,840$ transitions
- **Mini-batch size**: $163,840 / 32 = 5,120$ transitions
- **Number of mini-batches per epoch**: 32
- **Gradient updates per batch**: 4 full epochs over the collected data
- **Total updates per collect phase**: $32 \times 4 = 128$ SGD steps
- **Effective batch size per SGD step**: 5,120 transitions

**Training Schedule**:
- **Total training timesteps**: 200 million environment steps
- **Number of policy gradient updates**: $\approx 200M / 163,840 \approx 1,220$ collect-and-update cycles
- **Training time on A100 GPU**: Approximately 20-30 hours
- **Evaluation frequency**: Every collect phase (10 evaluations total during training)
- **Evaluation horizon**: 128 parallel environments running deterministically (no sensor noise) for 1000 steps

**Mini-batch Sampling**:
- Mini-batches are sampled uniformly at random (without replacement) from collected trajectories
- Sequential experience from the same trajectory may or may not be grouped together (depends on sampling order)
- This randomization decorrelates the mini-batches and stabilizes gradient estimates

#### JAX/Flax Implementation Details

The training is implemented using:
- **Framework**: JAX (functional array programming) with Flax neural network library
- **Computation**: JIT-compiled PyTree operations for efficient GPU utilization
- **Vectorization**: `jax.vmap` for parallelizing across 8192 environments
- **Randomness**: JAX PRNG (pseudo-random number generator) with explicit key splitting for reproducibility

**Specific implementation patterns**:
- Policy network weights are stored in Flax `nn.Module` with frozen-dict immutable state
- All forward passes (policy, value) are pure functions without side effects
- Gradients computed via `jax.grad()` with respect to network parameters
- SGD updates via Adam optimizer (Optax library) with learning rate $\eta$
- Loss computation is differentiable and composed: $L_{\text{total}} = -L^{\text{PPO}} + c_v L^V - c_e H$

#### Environment-Specific Configuration for Go1JoystickFlatTerrain

The configuration for the joystick control task is specialized relative to the default:

| Parameter | Default | Go1Joystick | Change | Reason |
|---|---|---|---|---|
| **Policy hidden sizes** | (128, 128, 128, 128) | (512, 256, 128) | Larger, fewer layers | Balance capacity and expressiveness for complex locomotion |
| **Value hidden sizes** | (256, 256, 256, 256, 256) | (512, 256, 128) | Consistent with policy | Deeper value network helps with privileged state processing |
| **Policy obs key** | "state" | "state" | Same | Policy uses only onboard observations |
| **Value obs key** | "state" | "privileged_state" | Asymmetric | Value network exploits simulation-only privileged information |
| **Total timesteps** | 100M | 200M | +100% | Joystick task requires longer training for gait refinement |
| **Learning rate** | $3e-4$ | $3e-4$ | Same | Standard for all Go1 tasks |
| **Entropy cost** | $1e-2$ | $1e-2$ | Same | Moderate exploration regularization |
| **Discounting** | 0.97 | 0.97 | Same | Short-horizon reward discounting |
| **Clipping epsilon** | (not set, uses default 0.2) | 0.2 | Same | Standard PPO clipping |
| **Num envs (train)** | 8192 | 8192 | Same | Fixed for computational efficiency |
| **Num envs (eval)** | 128 | 128 | Same | Evaluation on subset of configs |

The policy hidden layer configuration $(512, 256, 128)$ represents a gradually decreasing capacity bottleneck: information flow contracts from 512 → 256 → 128 → 12 (action dim), allowing the network to learn hierarchical feature representations. The value network $(512, 256, 128)$ has the same structure despite receiving a larger input (123 vs 48 dims), implemented via an input projection layer within the network.

#### Inference and Deployment Configuration

At deployment/inference time, the trained policy is used in deterministic mode:
$$a_t = \mu_\theta(o_t) \quad \text{(mean action, no sampling)}.$$
The variance/exploration is disabled, yielding crisp control outputs. This deterministic policy is used for:
- Real-robot rollouts (teleoperation, autonomous tasks)
- Evaluation episodes (reported success metrics)
- Video recording and qualitative assessment

Alternative: **stochastic inference** with sampled actions $a_t \sim \pi_\theta(\cdot \mid o_t)$ can be used for uncertainty quantification or to maintain exploratory behavior during learning.

### Complete Training Loop Schematic

The complete PPO training loop for Go1 locomotion is:

1. **Data Collection** (per environment, parallel across 8192):
   - Sample $o_t$ from current state
   - Compute action: $a_t \sim \pi_\theta(a_t | o_t)$ (stochastic sampling during training)
   - Execute in MuJoCo: $s_{t+1}, r_t = \text{Env.step}(s_t, a_t)$
   - Compute next observation: $o_{t+1}$
   - Store $(o_t, a_t, r_t, o_t^v, d_t)$ in trajectory buffer

2. **Trajectory Processing** (every 20 steps):
   - Compute value targets: $V_{\text{target}} = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$
   - Estimate advantages via GAE: $\hat{A}_t = V_{\text{target}} - V_\phi(o_t^v)$
   - Normalize advantages: $\hat{A}_t \leftarrow (\hat{A}_t - \mu_A) / (\sigma_A + 1e-5)$
   - Collect $163,840$ transitions total

3. **Policy Gradient Updates** (4 epochs, 32 minibatches/epoch):
   - Sample minibatch: $\{(o_t, a_t, \hat{A}_t)\}$ of size 5,120
   - Compute policy loss: $L^{\text{PPO}} = -\min(\varrho_t \hat{A}_t, \text{clip}(\varrho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$
   - Compute value loss: $L^V = (V_\phi(o_t^v) - \hat{R}_t)^2$
   - Compute entropy: $H = \text{mean}(\log \sigma_i)$
   - Compute total loss: $L = -L^{\text{PPO}} + c_v L^V - c_e H$
   - Backprop: $\theta \leftarrow \theta - \eta \nabla_\theta L$, $\phi \leftarrow \phi - \eta \nabla_\phi L$
   - Clip gradients: $\|\nabla L\|_2 \leq 1.0$

4. **Evaluation** (every 163,840 transitions):
   - Run 128 parallel episodes with deterministic policy (no noise)
   - Compute mean episode reward
   - Log metrics to TensorBoard/Weights&Biases

5. **Checkpoint Saving**:
   - Save weights every 10 evaluations (when performance improves)
   - Final checkpoint at 200M timesteps

#### Relation Between Network Depth, Width, and Learning Rate

The three hyperparameters interact as follows:

- **Wider networks** (512 vs 256 hidden units) increase model capacity, enabling learning of more complex policies. However, they require careful learning rate tuning to avoid overfitting or instability.
  
- **Deeper networks** (more hidden layers) allow hierarchical feature learning but increase the risk of vanishing/exploding gradients. Tanh activations with orthogonal initialization mitigate this.

- **Higher learning rates** ($\eta = 3e-4$ vs lower) accelerate convergence but can cause policy oscillations or divergence, especially if the loss landscape is non-smooth. Gradient clipping (max norm 1.0) provides a safety mechanism.

**For Go1 joystick task**: The $(512, 256, 128)$ architecture with $\eta = 3e-4$ was tuned empirically to balance:
- **Expressiveness**: 512 hidden units in policy can model complex locomotion behaviors
- **Stability**: Decreasing layer widths (512 → 256 → 128) create a bottleneck that encourages feature abstraction
- **Convergence speed**: $3e-4$ learning rate achieves good convergence in 200M steps (~24 hours on A100)
- **Generalization**: Moderate capacity (compared to e.g., 1024 units) reduces overfitting to training noise/domain randomization

### Domain Randomization for Robustness

To improve robustness to real-world variations and sim-to-real transfer, domain randomization is applied during training. Physical parameters are randomly perturbed in each parallel environment:

| Parameter | Randomization | Purpose |
|---|---|---|
| Floor friction coefficient | $\mu \sim \mathcal{U}(0.4, 1.0)$ | Account for terrain variation |
| Joint friction (damping) | scale $ \times \mathcal{U}(0.9, 1.1)$ | Vary dissipative losses |
| Joint armature (rotor inertia) | scale $ \times \mathcal{U}(1.0, 1.05)$ | Slight variance in motor properties |
| Torso CoM position | $+\mathcal{U}(-0.05, 0.05)$ m (x,y,z) | Account for packing CG uncertainty |
| All link masses | scale $ \times \mathcal{U}(0.9, 1.1)$ | Payload variation (±10%) |
| Torso mass only | $+\mathcal{U}(-1.0, 1.0)$ kg | Added external payload variation (±1 kg) |
| Joint home position offset | $+\mathcal{U}(-0.05, 0.05)$ rad (per joint) | Servo calibration drift |

These randomizations occur at the model level (per-episode in parallel environments) and are implemented via MuJoCo model transforms before physics simulation begins.

### Reward Function

The reward at each control step is defined as
$$r_t
=
\text{clip}\left(
\Delta t \sum_i w_i \rho_i(t),
\ 0,\ 10000
\right),$$
where $\rho_i(t)$ is an individual reward or penalty term (dimensionless or in reward units), $w_i \in \mathbb{R}$ is its scalar weight, and $\Delta t=0.02$ s is the control timestep. The clipping ensures bounded rewards for numerical stability. The 14 reward terms are:

**Tracking Terms:**
$$\rho_{\text{track,lin}}
=
\exp\left(
-\frac{(v_x^{\text{cmd}}-v_x)^2 + (v_y^{\text{cmd}}-v_y)^2}{\sigma}
\right),
\qquad
w_{\text{track,lin}} = 1.0$$

$$\rho_{\text{track,yaw}}
=
\exp\left(
-\frac{(\omega_z^{\text{cmd}}-\omega_z)^2}{\sigma}
\right),
\qquad
w_{\text{track,yaw}} = 0.5$$

**Base Motion Penalties:**
$$\rho_{\text{lin-z}} = v_z^2,
\qquad
w_{\text{lin-z}} = -0.5$$

$$\rho_{\text{ang-xy}} = \omega_x^2 + \omega_y^2,
\qquad
w_{\text{ang-xy}} = -0.05$$

$$\rho_{\text{orient}} = u_x^2 + u_y^2,
\qquad
w_{\text{orient}} = -5.0$$

**Posture Terms:**
$$\rho_{\text{pose}}
=
\exp\left(
-\sum_{i=1}^{12}\alpha_i (q_{j,i}-q_{j,i}^{\text{home}})^2
\right),
\qquad
w_{\text{pose}} = 0.5$$
with
$$\alpha = [1,1,0.1,1,1,0.1,1,1,0.1,1,1,0.1].$$

$$\rho_{\text{stand}}
=
\left\|q_j - q_j^{\text{home}}\right\|_1 \, \mathbf{1}\{\|c_t\|<0.01\},
\qquad
w_{\text{stand}} = -1.0$$

**Termination and Joint Limits:**
$$\rho_{\text{term}} = d_t,
\qquad
w_{\text{term}} = -1.0$$

$$\rho_{\text{limit}}
=
\sum_{i=1}^{12}
\left(
[\underline{q}^{\,s}_i - q_{j,i}]_+ + [q_{j,i}-\bar{q}^{\,s}_i]_+
\right),
\qquad
w_{\text{limit}} = -1.0$$
where $[x]_+ = \max(x,0)$ and $\underline{q}^{\,s}_i,\bar{q}^{\,s}_i$ are soft joint limits set to 95% of the physical joint range.

**Energy Penalties:**
$$\rho_{\tau}
=
\sqrt{\sum_{i=1}^{12}\tau_i^2} + \sum_{i=1}^{12}|\tau_i|,
\qquad
w_{\tau} = -0.0002$$

$$\rho_{\text{act}}
=
\|a_t-a_{t-1}\|_2^2,
\qquad
w_{\text{act}} = -0.01$$

$$\rho_{\text{energy}}
=
\sum_{i=1}^{12} |\dot{q}_{j,i}|\,|\tau_i|,
\qquad
w_{\text{energy}} = -0.001$$

**Foot-Related Terms:**
$$\rho_{\text{slip}}
=
\sum_{k=1}^{4}
\|v^{\text{foot}}_{xy,k}\|_2^2 \, m_k \, \mathbf{1}\{\|c_t\|>0.01\},
\qquad
w_{\text{slip}} = -0.1$$
where $m_k \in \{0,1\}$ indicates foot-ground contact.

$$\rho_{\text{clr}}
=
\sum_{k=1}^{4}
|z^{\text{foot}}_k - h^\star|
\sqrt{\|v^{\text{foot}}_{xy,k}\|_2},
\qquad
w_{\text{clr}} = -2.0$$
with target swing-foot height
$$h^\star = 0.1 \text{ m}.$$

$$\rho_{\text{fh}}
=
\sum_{k=1}^{4}
\left(
\frac{h^{\text{peak}}_k}{h^\star} - 1
\right)^2
\delta_k \,
\mathbf{1}\{\|c_t\|>0.01\},
\qquad
w_{\text{fh}} = -0.2$$
where $h_k^{\text{peak}}$ is the maximum swing height of foot $k$ and $\delta_k$ indicates first ground contact after a swing phase.

$$\rho_{\text{air}}
=
\sum_{k=1}^{4}
(T_k^{\text{air}} - 0.1)\,\delta_k \,
\mathbf{1}\{\|c_t\|>0.01\},
\qquad
w_{\text{air}} = 0.1$$

### Summary and Integration

The tracking terms ($\rho_{\text{track,lin}}, \rho_{\text{track,yaw}}$) encourage the robot to follow commanded forward, lateral, and yaw velocities, forming the primary task objective. These use Gaussian kernels with tracking $\sigma = 0.25$ to reward close tracking over perfect tracking.

The base motion penalties ($\rho_{\text{lin-z}}, \rho_{\text{ang-xy}}, \rho_{\text{orient}}$) constrain unwanted motions: vertical drift, sideways tilting, and forward/backward tilting respectively, ensuring stable upright locomotion.

The posture terms ($\rho_{\text{pose}}, \rho_{\text{stand}}$) maintain the robot near a nominal standing configuration with weighted per-joint penalties (lower weight for lateral motions which have less stability contribution). The stand-still term penalizes posture deviation when no command is given.

The termination penalty ($\rho_{\text{term}}$) flags episode-ending falls, providing immediate negative signal for instability. Joint limit penalties ($\rho_{\text{limit}}$) prevent unsafe configurations by penalizing soft limits at 95% of mechanical range.

Energy penalties ($\rho_{\tau}, \rho_{\text{act}}, \rho_{\text{energy}}$) reduce aggressive, jerky, or power-hungry control. Torque penalty uses both L2 and L1 norms; action-rate penalizes accelerations; energy combines joint velocity and torque magnitudes.

The foot-related terms constitute a gait-shaping subsystem:
- **Slip penalty** ($\rho_{\text{slip}}$): Slipping feet consume energy; active only when moving ($ \|c_t\| > 0.01$)
- **Clearance penalty** ($\rho_{\text{clr}}$): Swing-leg feet should maintain target height $h^\star = 0.1$ m to avoid stumbling
- **Foot height penalty** ($\rho_{\text{fh}}$): Overly high swings waste energy and destabilize; penalizes deviation from target height at touchdown
- **Air time reward** ($\rho_{\text{air}}$): Positive reward for swing phase, encouraging liftoff and clear ground clearance, only when moving

Together, the reward function embeds domain knowledge about quadruped locomotion into a learnable optimization landscape. The weighting vector $\mathbf{w} = [w_1, \ldots, w_{14}]^\top$ balances competing objectives, allowing the trainable policy to discover energy-efficient, stable, command-tracking gaits.

**Overall Training Loop**: At each PPO update step, trajectories are collected from 8192 parallel environments over 20 timesteps. Rewards are accumulated and discounted; advantages are estimated from value function; policy and value networks are updated via gradient descent (with clipping) over 4 epochs of 32 mini-batches each. The process repeats for 200M environment steps, with periodic evaluation on unseen command sequences. The learned policy $\pi_\theta$ outputs smooth, continuous actions converted to joint positions by the PD controller, producing coordinated quadruped locomotion.
