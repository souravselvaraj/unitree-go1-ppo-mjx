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

### Rough Terrain Environment

A second terrain variant, `Go1JoystickRoughTerrain`, replaces the flat plane with a procedurally generated heightfield loaded from a PNG image (`go1_ppo/assets/hfield.png`). The terrain is modeled as a MuJoCo `hfield` geom with world dimensions

$$\text{width} \times \text{height} = 10 \times 10 \text{ m}, \quad h_{\max} = 0.05 \text{ m}, \quad \text{base depth} = 1.0 \text{ m}.$$

The elevation at each grid cell is linearly mapped from the 8-bit pixel intensity (0–255) of the PNG image to the range $[0, h_{\max}]$. The surface uses a photorealistic rocky texture (Poly Haven "rock\_face") tiled at $5 \times 5$ repeats per 10 m tile.

#### Floor Contact Properties

The rough terrain floor geom is assigned a higher static friction coefficient
$$\mu_{\text{rough}} = 1.0$$
compared to $\mu_{\text{flat}} = 0.6$ on flat terrain, reflecting the greater grip of natural rocky surfaces and the geometric interlocking of foot contact points.

#### Contact Solver Configuration

Navigating rough terrain generates substantially more simultaneous contacts per environment step because surface normal variation causes multi-point foot–ground interactions. The MJX solver limits are therefore expanded:

| Parameter | Flat Terrain | Rough Terrain |
|---|---|---|
| `nconmax` | $4 \times 8192 = 32{,}768$ | $8 \times 8192 = 65{,}536$ |
| `njmax` | 40 | 60 |

These limits are applied automatically in `Joystick.__init__` when the task name begins with `"rough"`. Insufficient `nconmax` causes silent contact dropping in MJX, leading to unphysical foot penetration and corrupted gradients during training; doubling the limit ensures all foot–terrain contacts are resolved at each substep.

#### Terrain Generation and Geometry

The heightfield PNG encodes terrain as a single-channel grayscale image. MuJoCo interprets the pixel grid as a rectangular array of elevation samples and tessellates it into triangle meshes for contact computation. Key geometric properties:

- **Horizontal resolution**: determined by the PNG image dimensions (pixels per 10 m); finer resolution produces more detailed surface features.
- **Maximum step height**: $h_{\max} = 0.05$ m (5 cm), below the nominal foot clearance target $h^\star = 0.1$ m, so individual steps are traversable but create meaningful perturbations.
- **Base depth** (below minimum elevation): 1.0 m, ensuring the geom occupies sufficient volume to prevent tunneling at terrain edges.

The robot home keyframe places the torso at $z = 0.35$ m above the heightfield reference plane (identical to flat terrain), so at episode reset the feet land at varying heights depending on spawn location within the 10 m tile.

#### Gait Challenges and Behavioral Differences

On rough terrain the robot must adapt its gait to four additional sources of difficulty absent on flat ground:

1. **Variable foot contact heights**: ground reaction forces vary in direction and magnitude as each foot lands on differently sloped patches, exciting roll and pitch moments at the torso.
2. **Consistent clearance under uncertainty**: the clearance term $\rho_{\text{clr}}$ and foot-height term $\rho_{\text{fh}}$ retain target height $h^\star = 0.1$ m, but achieving this requires active foot lift against surface geometry that is not directly observed by the policy.
3. **Increased orientation disturbances**: slope-induced moments make the orientation penalty $\rho_{\text{orient}}$ (weight $-5.0$) more frequently non-zero, demanding stronger active stabilization.
4. **Foot slip on inclined patches**: inclined surface patches create lateral components of ground reaction force; the slip penalty $\rho_{\text{slip}}$ remains active but is harder to minimize due to surface tilt.

The reward function is identical to the flat-terrain variant — no terrain-specific shaping terms are added. The policy therefore implicitly learns to navigate uneven ground through the same multi-objective reward signal, relying on domain randomization (particularly floor friction $\mu \sim \mathcal{U}(0.4, 1.0)$) and the variety of hfield contact geometry to generalize.

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

#### Actor State Vector (Policy Input)

The policy $\pi_\theta$ receives only observations that are realizable on the physical robot from onboard sensors, making the trained controller directly deployable without any simulation-only state. The actor observation is a 48-dimensional vector
$$o_t =
\bigl[
\tilde{v}_t^b,\;
\tilde{\omega}_t^b,\;
\tilde{g}_t^b,\;
\tilde{q}_{j,t} - q_j^{\text{home}},\;
\widetilde{\dot{q}}_{j,t},\;
a_{t-1},\;
c_t
\bigr]^\top
\in \mathbb{R}^{48}.$$

Each field is described in detail below:

| Index range | Field | Dims | Source | Noise added | Notes |
|:---:|---|:---:|---|:---:|---|
| 0–2 | $\tilde{v}_t^b$ — body-frame linear velocity | 3 | `local_linvel` sensor (MJX computed) | $\pm 0.1$ m/s | Expressed in body frame $B$. Not directly observable on real hardware (estimated via VIO or state estimator). |
| 3–5 | $\tilde{\omega}_t^b$ — angular velocity | 3 | Gyroscope (`gyro` sensor) | $\pm 0.2$ rad/s | Body frame, measured by onboard IMU gyroscope. Directly available on real Go1. |
| 6–8 | $\tilde{g}_t^b$ — gravity vector in body frame | 3 | Inclinometer (`gravity` derived from orientation) | $\pm 0.05$ (mag.) | Unit gravity direction expressed in body frame; encodes roll and pitch angles. $\tilde{g}_t^b \approx R_{WB}^\top [0,0,-1]^\top$ in noise-free case. |
| 9–20 | $\tilde{q}_{j,t} - q_j^{\text{home}}$ — joint position offsets | 12 | Joint encoders (`qpos[7:]`) | $\pm 0.03$ rad | Per-joint deviation from home pose. Order: FR (abd, hip, knee), FL, RR, RL. |
| 21–32 | $\widetilde{\dot{q}}_{j,t}$ — joint velocities | 12 | Joint velocity (`qvel[6:]`) | $\pm 1.5$ rad/s | Per-joint angular velocity. High noise scale reflects encoder differentiation noise. |
| 33–44 | $a_{t-1}$ — previous action | 12 | Policy output at $t-1$ | None | Provides temporal context; lets policy estimate current joint state residual and plan smooth continuations. |
| 45–47 | $c_t$ — locomotion command | 3 | Command sampler / teleoperation | None | $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}]^\top$; no noise injected since command is internally known. |

**Observation normalization**: During training, observations are normalized by a running mean and variance (`normalize_observations=True`), updated online across all 8192 parallel environments. This improves gradient conditioning and ensures that all 48 dimensions contribute comparably to the loss. The normalization statistics are saved with the checkpoint and applied identically at deployment.

**Design rationale**: The 48-dimensional state deliberately excludes all non-onboard information (foot contact forces, ground truth velocity, world-frame pose). The previous action $a_{t-1}$ substitutes for an explicit foot contact indicator — a policy that learns to maintain consistent action sequences implicitly tracks stance/swing phases through the action history. The gravity vector $\tilde{g}_t^b$ encodes orientation without requiring a full quaternion, and its body-frame expression is invariant to yaw rotation, which matches the yaw-agnostic nature of the locomotion task.

Uniform sensor noise is injected during training via
$$\tilde{x}_t = x_t + \sigma_{\text{level}} \cdot n_{\text{scale}} \cdot u_t,$$
where $x_t$ is the true measurement, $\sigma_{\text{level}} = 1.0$ is the noise level multiplier, $n_{\text{scale}}$ is the modality-specific scale factor, and $u_t \sim \mathcal{U}(-1, 1)$ is uniform random noise. Modality-specific noise scales are:

| Sensor | Scale $n_{\text{scale}}$ | Physical interpretation |
|--------|----------|---|
| Joint position | 0.03 rad | $\approx 1.7°$ peak error; typical magnetic encoder quantization + hysteresis |
| Joint velocity | 1.5 rad/s | Differentiated encoder signal; large noise reflects velocity estimation from position |
| Gyroscope | 0.2 rad/s | IMU rate noise; 3-sigma corresponds to $\approx 11°$/s |
| Inclinometer (gravity) | 0.05 (normalized magnitude) | Tilt angle error $\approx 2.9°$; accelerometer-derived tilt noise |
| Linear velocity | 0.1 m/s | State estimator velocity error; represents VIO drift |

This noise helps the learned policy generalize to real-world sensor measurements and reduces overfitting to noise-free simulation.

#### Critic State Vector (Privileged Input)

An asymmetric actor-critic structure is employed. The policy $\pi_\theta$ receives only the 48-dimensional $o_t$ and operates with onboard-realizable observations (compatible with real robot deployment). The critic $V_\phi$ receives a privileged 123-dimensional state vector
$$o_t^v = \bigl[o_t,\; s_t^{\text{priv}}\bigr]^\top \in \mathbb{R}^{123},$$
where $s_t^{\text{priv}} \in \mathbb{R}^{75}$ is simulation-only privileged information appended after $o_t$.

The privileged fields are:

| Index range (in $o_t^v$) | Field | Dims | Source | Notes |
|:---:|---|:---:|---|---|
| 0–47 | $o_t$ (actor state, as above) | 48 | Same as actor input | Noisy observations identical to what the policy sees |
| 48–50 | Clean gyroscope $\omega^b$ | 3 | `gyro` (no noise) | Noise-free IMU angular velocity; helps critic estimate accurate body dynamics |
| 51–53 | Accelerometer $a^b$ | 3 | `accelerometer` (no noise) | Linear acceleration in body frame; not in actor state at all |
| 54–56 | Clean gravity vector $g^b$ | 3 | Gravity (no noise) | Noise-free gravity direction; removes inclinometer drift from critic's orientation estimate |
| 57–59 | Clean body-frame linear velocity $v^b$ | 3 | `local_linvel` (no noise) | Noise-free velocity for accurate reward/advantage computation |
| 60–62 | Global linear velocity $v^W$ | 3 | `global_linvel` | World-frame velocity; unavailable on real robot (no GPS/mocap) |
| 63–65 | Global angular velocity $\omega^W$ | 3 | `global_angvel` | World-frame angular velocity; complements body-frame gyro |
| 66–77 | Clean joint positions $q_j - q_j^{\text{home}}$ | 12 | `qpos[7:]` (no noise) | Noise-free offsets; removes 0.03 rad encoder noise from critic's state estimate |
| 78–89 | Clean joint velocities $\dot{q}_j$ | 12 | `qvel[6:]` (no noise) | Noise-free velocities; removes 1.5 rad/s differentiation noise |
| 90–101 | Actuator forces $\tau$ | 12 | `actuator_force` | PD torques actually applied; not directly sensed by policy |
| 102–105 | Foot contact states $m_k$ | 4 | Touch sensors (binary) | Per-foot contact boolean; $m_k \in \{0,1\}$ |
| 106–117 | Foot linear velocities $v^{\text{foot}}_k$ | 12 | `FEET_SITES` global linvel sensors | 3D foot velocity for each of 4 feet (FR, FL, RR, RL) |
| 118–121 | Foot air times $T_k^{\text{air}}$ | 4 | Accumulated `feet_air_time` | Time since last contact per foot; accumulated and reset at touchdown |
| 122–124 | Perturbation force on torso | 3 | `xfrc_applied[torso, :3]` | External force applied during perturbation training; zero if pert disabled |
| 125 | Perturbation active indicator | 1 | `steps_since_last_pert >= steps_until_next_pert` | Boolean: 1 if perturbation is currently scheduled; helps critic predict disturbed dynamics |

**Total**: $48 + 75 = 123$ dimensions. Note that the foot velocity field is 12-dimensional (3D per foot) in `privileged_state`, while only 8 of those dimensions (xy components) were listed in the original high-level summary; the full 3D velocity including $z$ is included.

**Design rationale for privileged state**: The critic's role during training is to estimate the expected return $V_\phi(o_t^v)$ as accurately as possible for computing GAE advantages. Providing noise-free sensors and simulation-only quantities (actuator forces, foot contacts, perturbation state) directly improves the accuracy of these value estimates without requiring the policy to observe them. The asymmetry is a form of privileged distillation: the critic learns from richer information, generating better advantage signals that guide the policy toward better behavior — even though the policy itself will never see those privileged inputs at deployment.

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

#### Reward Term Reference Table

The full set of 16 reward and penalty terms is summarized below, ordered by functional group. All terms are evaluated at each control step; terms marked "conditional" are multiplied by an indicator that zeroes them when the locomotion command is near zero ($\|c_t\| \leq 0.01$ rad/s or m/s).

| Symbol | Name | Weight $w_i$ | Formula (unnormalized) | Conditional | Purpose |
|--------|------|:---:|---|:---:|---|
| $\rho_{\text{track,lin}}$ | Lin. vel. tracking | $+1.0$ | $\exp\!\left(-\frac{\|v_{xy}^{\text{cmd}}-v_{xy}\|^2}{\sigma}\right)$, $\sigma{=}0.25$ | No | Primary task: match forward/lateral speed |
| $\rho_{\text{track,yaw}}$ | Yaw rate tracking | $+0.5$ | $\exp\!\left(-\frac{(\omega_z^{\text{cmd}}-\omega_z)^2}{\sigma}\right)$, $\sigma{=}0.25$ | No | Primary task: match yaw rate |
| $\rho_{\text{lin-z}}$ | Vertical vel. | $-0.5$ | $v_z^2$ | No | Suppress vertical bouncing |
| $\rho_{\text{ang-xy}}$ | Roll/pitch rate | $-0.05$ | $\omega_x^2+\omega_y^2$ | No | Suppress trunk rocking |
| $\rho_{\text{orient}}$ | Trunk tilt | $-5.0$ | $u_x^2+u_y^2$ (up-vector xy deviation) | No | Strongly penalize non-upright posture |
| $\rho_{\text{pose}}$ | Joint pose | $+0.5$ | $\exp\!\bigl(-\sum_i \alpha_i(q_i-q_i^{\text{home}})^2\bigr)$ | No | Stay near nominal standing pose |
| $\rho_{\text{stand}}$ | Stand still | $-1.0$ | $\sum_i|q_i-q_i^{\text{home}}|$ | When $\|c_t\|<0.01$ | No fidgeting when no command |
| $\rho_{\text{term}}$ | Termination | $-1.0$ | $d_t \in \{0,1\}$ (fall indicator) | No | Immediate penalty for episode-ending fall |
| $\rho_{\text{limit}}$ | Joint limits | $-1.0$ | $\sum_i[\underline{q}^s_i - q_i]_+ + [q_i - \bar{q}^s_i]_+$ | No | Stay within 95% of mechanical range |
| $\rho_{\tau}$ | Torque | $-0.0002$ | $\sqrt{\sum\tau_i^2}+\sum|\tau_i|$ (L2+L1) | No | Reduce peak and sustained torque use |
| $\rho_{\text{act}}$ | Action rate | $-0.01$ | $\sum_i(a_{t,i}-a_{t-1,i})^2$ | No | Penalize rapid action changes (jerk proxy) |
| $\rho_{\text{energy}}$ | Energy | $-0.001$ | $\sum_i|\dot{q}_i||\tau_i|$ | No | Penalize mechanical power consumption |
| $\rho_{\text{slip}}$ | Foot slip | $-0.1$ | $\sum_k\|v_{xy,k}^{\text{foot}}\|^2 m_k$ | Yes | Penalize lateral velocity of grounded feet |
| $\rho_{\text{clr}}$ | Foot clearance | $-2.0$ | $\sum_k|z_k^{\text{foot}}-h^\star|\sqrt{\|v_{xy,k}^{\text{foot}}\|}$ | No | Feet should swing at $h^\star{=}0.1$ m |
| $\rho_{\text{fh}}$ | Swing peak height | $-0.2$ | $\sum_k\!\left(\frac{h_k^{\text{peak}}}{h^\star}-1\right)^2\!\delta_k$ | Yes | Normalize peak swing height to $h^\star$ |
| $\rho_{\text{air}}$ | Foot air time | $+0.1$ | $\sum_k(T_k^{\text{air}}-0.1)\delta_k$ | Yes | Encourage sufficient stance–swing cycle |

**Notes on key design choices:**
- **Tracking kernel**: the Gaussian form $\exp(-e^2/\sigma)$ gives reward 1 at perfect tracking ($e=0$), reward $e^{-1}\!\approx 0.37$ at error $\sqrt{\sigma}=0.5$ m/s, and approaches 0 for large errors. This saturates the gradient at large errors to avoid destabilizing updates.
- **Clearance weighting**: $\rho_{\text{clr}}$ multiplies the height error by $\sqrt{\|v_{xy}\|}$ rather than a contact indicator, so it penalizes clearance violations specifically while the foot is moving horizontally — it is automatically silent for a stationary foot and largest during mid-swing.
- **Torque L2+L1**: the combined $\sqrt{\sum\tau^2} + \sum|\tau|$ penalizes both peak torque (via L2) and overall torque magnitude (via L1), discouraging both brief spikes and sustained high torques.
- **Action rate as jerk proxy**: $\rho_{\text{act}} = \sum(a_t - a_{t-1})^2$ penalizes the first-order finite difference of the action. Since actions map to position targets, this approximates joint velocity overshoot rather than true mechanical jerk.
- **Pose weighting**: the per-joint weight vector $\alpha = [1,1,0.1]\times 4$ downweights knee joints (index 3 per leg, $\alpha=0.1$) relative to hip and abduction joints ($\alpha=1.0$), allowing more knee variation during locomotion while keeping the upper leg close to the nominal configuration.
- **Stand-still vs. pose**: $\rho_{\text{stand}}$ activates only at zero command (no movement requested) and uses the L1 norm to keep all joints near home, while $\rho_{\text{pose}}$ is always active and uses a Gaussian (soft) reward — together they prevent both idling drift and aggressive posture deviation during locomotion.

#### Reward Scaling and Total Magnitude

The total per-step reward is
$$r_t = \text{clip}\!\left(\Delta t \sum_{i=1}^{16} w_i\,\rho_i(t),\ 0,\ 10{,}000\right), \qquad \Delta t = 0.02 \text{ s}.$$

Multiplying by $\Delta t$ converts the weighted sum from a per-step basis to a per-second basis, making the reward roughly unit-consistent across different control frequencies. In practice, a well-behaved policy achieves:
- Tracking terms contributing approximately $\Delta t \times (1.0\cdot 0.9 + 0.5\cdot 0.9) \approx 0.027$ per step (near-perfect tracking, $\approx 1.35$ per second).
- Energy and regularization terms contributing small negative values ($\lesssim -0.005$ per step at typical operation).
- The net reward per step is thus dominated by tracking performance, ensuring that command following is the primary optimization target.

The lower clip at 0 is primarily a safety measure against rare numerical instability; normal operation never triggers it because the positive tracking terms dominate. The upper clip at 10,000 is never reached in practice.

### Summary and Integration

The tracking terms ($\rho_{\text{track,lin}}, \rho_{\text{track,yaw}}$) encourage the robot to follow commanded forward, lateral, and yaw velocities, forming the primary task objective. These use Gaussian kernels with tracking $\sigma = 0.25$ to reward close tracking over perfect tracking.

The base motion penalties ($\rho_{\text{lin-z}}, \rho_{\text{ang-xy}}, \rho_{\text{orient}}$) constrain unwanted motions: vertical drift, sideways tilting, and forward/backward tilting respectively, ensuring stable upright locomotion.

The posture terms ($\rho_{\text{pose}}, \rho_{\text{stand}}$) maintain the robot near a nominal standing configuration with weighted per-joint penalties (lower weight for knee joints which need more range during locomotion). The stand-still term penalizes posture deviation only when no command is given.

The termination penalty ($\rho_{\text{term}}$) flags episode-ending falls, providing immediate negative signal for instability. Joint limit penalties ($\rho_{\text{limit}}$) prevent unsafe configurations by penalizing soft limits at 95% of mechanical range.

Energy penalties ($\rho_{\tau}, \rho_{\text{act}}, \rho_{\text{energy}}$) reduce aggressive, jerky, or power-hungry control. Torque penalty uses both L2 and L1 norms to suppress both peak torques and sustained high torque; action-rate penalizes the squared first-order difference of consecutive actions as a proxy for joint jerk; energy penalizes mechanical power $\sum|\dot{q}_i||\tau_i|$ directly.

The foot-related terms constitute a gait-shaping subsystem:
- **Slip penalty** ($\rho_{\text{slip}}$): Slipping feet consume energy; active only when moving ($\|c_t\| > 0.01$)
- **Clearance penalty** ($\rho_{\text{clr}}$): Swing-leg feet should maintain target height $h^\star = 0.1$ m; weighted by $\sqrt{\|v_{xy}\|}$ so it is silent for stationary feet and maximal during mid-swing
- **Foot height penalty** ($\rho_{\text{fh}}$): Penalizes the squared relative deviation of the peak swing height from $h^\star$ at the moment of touchdown; discourages both under- and over-swinging
- **Air time reward** ($\rho_{\text{air}}$): Positive reward proportional to air time in excess of 0.1 s, encouraging liftoff and clear ground clearance, active only when moving

Together, the reward function embeds domain knowledge about quadruped locomotion into a learnable optimization landscape. The weighting vector $\mathbf{w} = [w_1, \ldots, w_{16}]^\top$ balances competing objectives, allowing the trainable policy to discover energy-efficient, stable, command-tracking gaits.

**Overall Training Loop**: At each PPO update step, trajectories are collected from 8192 parallel environments over 20 timesteps. Rewards are accumulated and discounted; advantages are estimated from value function; policy and value networks are updated via gradient descent (with clipping) over 4 epochs of 32 mini-batches each. The process repeats for 200M environment steps, with periodic evaluation on unseen command sequences. The learned policy $\pi_\theta$ outputs smooth, continuous actions converted to joint positions by the PD controller, producing coordinated quadruped locomotion.
