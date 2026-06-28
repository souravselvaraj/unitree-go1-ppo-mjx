# Unitree Go1 PPO: JAX/MJX Locomotion in MuJoCo

Train and teleoperate PPO locomotion policies for the Unitree Go1 quadruped with
JAX, Brax PPO, MuJoCo MJX, asymmetric actor-critic observations, domain
randomization, and flat or rough-terrain MJCF environments.

See [modeling_section.md](modeling_section.md) for the environment design,
reward terms, observation spaces, and training architecture.

Repository: [souravselvaraj/unitree-go1-ppo-mjx](https://github.com/souravselvaraj/unitree-go1-ppo-mjx.git)

## Demo

![Unitree Go1 PPO locomotion demo](docs/media/go1_locomotion_demo.gif)

Trimmed real-world Go1 locomotion demo from the clean walking segment (`00:00:07`-`00:00:27`).

## Why This Repo

- Unitree Go1 quadruped locomotion in MuJoCo MJX
- PPO training with JAX, Flax, Brax, and Orbax checkpoints
- Asymmetric actor-critic setup: onboard policy observations plus privileged critic state
- Domain randomization for friction, inertia, center of mass, and mass
- Gait shaping with foot clearance, slip, air-time, and energy penalties
- JAX/MJX vectorized training with `jax.vmap` for high-throughput simulation
- Flat and rough terrain tasks with vendored MJCF, STL, PNG, and heightfield assets
- Keyboard teleoperation for trained policies

## Environments

| Environment | Terrain | Use case |
|---|---|---|
| `Go1JoystickFlatTerrain` | Flat plane | Baseline Go1 velocity-command locomotion |
| `Go1JoystickRoughTerrain` | Heightfield rough terrain | Robust locomotion and contact-rich training |

## Setup

Requirements:

- Python 3.12
- `uv`
- NVIDIA GPU with CUDA 

Install:

```bash
git clone https://github.com/souravselvaraj/unitree-go1-ppo-mjx.git
cd unitree-go1-ppo-mjx
uv sync
```

The commands below use `uv run`, so activating `.venv` is optional.

## Training

Quick smoke test:

```bash
uv run python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True \
  --num_timesteps=10000 \
  --num_envs=256
```

Full flat-terrain training:

```bash
uv run python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True \
  --domain_randomization=True
```

Rough-terrain training:

```bash
uv run python training/train_jax_ppo.py \
  --env_name=Go1JoystickRoughTerrain \
  --use_tb=True \
  --domain_randomization=True
```

Training outputs are written under `logs/`, which is intentionally ignored by
git. Publish trained checkpoints separately through GitHub Releases, Hugging
Face, Google Drive, or another artifact store if you want others to reproduce a
specific policy.

## Monitoring

```bash
uv run tensorboard --logdir ./logs --port 6006
```

Open `http://localhost:6006` to view live training metrics.

## Teleoperation

Run a trained policy with keyboard control:

```bash
uv run python scripts/teleop_go1_keyboard.py \
  --run_dir ./logs/<Go1Joystick...run> \
  --env Go1JoystickRoughTerrain \
  --deterministic
```

Keyboard controls:

| Key | Action |
|---|---|
| `w` / `s` | Forward/backward velocity |
| `a` / `d` | Lateral velocity |
| `q` / `e` | Yaw rotation |
| `0` | Zero all commands |
| `[` / `]` | Slower/faster playback |
| `space` | Pause simulation |
| `r` | Reset episode |
| `x` | Quit |

## Project Structure

```text
.
├── go1_ppo/
│   ├── assets/          # Go1 meshes, terrain textures, heightfields
│   ├── configs/         # PPO and locomotion hyperparameters
│   ├── envs/            # Go1 MJX environment implementations
│   ├── scene/           # MuJoCo MJCF scene files
│   ├── registry.py      # Environment registration and loading
│   ├── mjx_env.py       # Shared MJX environment base
│   └── wrapper.py       # Brax training wrappers
├── training/            # Main PPO training entrypoint
├── scripts/             # Training and teleoperation utilities
├── modeling_section.md  # Detailed technical notes
├── pyproject.toml       # Python package metadata
└── uv.lock              # Reproducible dependency lockfile
```


## License

Original project code is licensed under Apache-2.0. Some files are adapted from
DeepMind/MuJoCo Apache-2.0 code, and the vendored Unitree Go1 model assets retain
their BSD-3-Clause license. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
