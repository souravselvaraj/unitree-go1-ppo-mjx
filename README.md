# go1_ppo

JAX/Flax-based PPO training and teleoperation for Unitree Go1 quadruped in MuJoCo. Includes asymmetric actor-critic learning, domain randomization, and multiple terrain types.

**See [modeling_section.md](modeling_section.md) for detailed environment, reward function, and training architecture documentation.**

## Environments

| Environment | Description |
|---|---|
| `Go1JoystickFlatTerrain` | Flat ground, ideal for baseline training |
| `Go1JoystickRoughTerrain` | Randomized heightfield with Gaussian noise, tests robustness |

## Setup

Requirements:

- Python 3.12
- `uv`
- NVIDIA GPU recommended for training

Install:

```bash
cd /home/sourav/skadi/go1_ppo
uv sync
source .venv/bin/activate
```

## Training

Quick test (10k steps):

```bash
python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True \
  --num_timesteps=10000 \
  --num_envs=256
```

Full training (200M steps):

```bash
python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True
```

Rough terrain training:

```bash
python training/train_jax_ppo.py \
  --env_name=Go1JoystickRoughTerrain \
  --use_tb=True
```

## Monitoring

```bash
tensorboard --logdir ./logs --port 6006
```

Open `http://localhost:6006` to view live training metrics.

## Teleoperation

Run trained policy with keyboard control:

```bash
python scripts/teleop_go1_keyboard.py \
  --run_dir ./logs/Go1JoystickRoughTerrain-20260425-153219/
```

The script will **prompt you to select a terrain**:

```
=== Available Environments ===
  1. Go1JoystickFlatTerrain
  2. Go1JoystickRoughTerrain

Select environment (1-2):
```

You can also skip the prompt with `--env`:

```bash
python scripts/teleop_go1_keyboard.py \
  --run_dir ./logs/Go1JoystickRoughTerrain-20260425-153219/ \
  --env Go1JoystickRoughTerrain \
  --deterministic
```

**Keyboard Controls:**

| Key | Action |
|---|---|
| `w`/`s` | Forward/backward velocity |
| `a`/`d` | Lateral velocity |
| `q`/`e` | Yaw rotation |
| `0` | Zero all commands |
| `[`/`]` | Slower/faster playback |
| `space` | Pause simulation |
| `r` | Reset episode |
| `x` | Quit |

## Project Structure

```
.
├── envs/               # Environment implementations
│   ├── joystick.py     # Main Joystick env class & reward function
│   ├── randomize.py    # Domain randomization
│   └── go1_constants.py # XML paths & task definitions
├── scene/              # MuJoCo MJCF scene files
├── training/           # PPO training loop
├── scripts/            # Utilities (teleop, evaluation)
├── configs/            # Hyperparameter configs
├── logs/               # Training checkpoints & TensorBoard events
├── assets/             # Heightfield images for rough terrain
├── registry.py         # Environment registration
├── mjx_env.py          # Base MjxEnv class
└── modeling_section.md # Detailed technical documentation
```

## Key Features


- **Gait Shaping**: Foot clearance, slip, air-time, and energy penalties
- **JAX/MJX**: Fully differentiable with `jax.vmap` for efficient 8192-parallel training
