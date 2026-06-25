# Unitree Go1 PPO: JAX/MJX Locomotion in MuJoCo

Train and teleoperate PPO locomotion policies for the Unitree Go1 quadruped with
JAX, Brax PPO, MuJoCo MJX, asymmetric actor-critic observations, domain
randomization, and flat or rough-terrain MJCF environments.

See [modeling_section.md](modeling_section.md) for the environment design,
reward terms, observation spaces, and training architecture.

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
- NVIDIA GPU with CUDA is recommended for serious training

Install:

```bash
git clone https://github.com/<your-user>/<repo-name>.git
cd <repo-name>
uv sync
source .venv/bin/activate
```

## Training

Quick smoke test:

```bash
python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True \
  --num_timesteps=10000 \
  --num_envs=256
```

Full flat-terrain training:

```bash
python training/train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_tb=True \
  --domain_randomization=True
```

Rough-terrain training:

```bash
python training/train_jax_ppo.py \
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
tensorboard --logdir ./logs --port 6006
```

Open `http://localhost:6006` to view live training metrics.

## Teleoperation

Run a trained policy with keyboard control:

```bash
python scripts/teleop_go1_keyboard.py \
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

## GitHub Search Setup

Recommended repository name:

```text
unitree-go1-ppo-mjx
```

Good shorter alternatives:

```text
go1-ppo-mjx
unitree-go1-ppo
```

Use this GitHub description:

```text
PPO locomotion training for the Unitree Go1 quadruped using JAX, Brax, MuJoCo MJX, and domain randomization.
```

Suggested GitHub topics:

```text
unitree-go1, quadruped-robot, legged-locomotion, reinforcement-learning,
ppo, mujoco, mjx, jax, brax, robotics, domain-randomization, sim-to-real
```

`go1_ppo` is a reasonable Python package name, but for the public GitHub repo a
hyphenated name with `unitree`, `go1`, `ppo`, and `mjx` will be easier to find.

## Publish Checklist

- Keep generated `logs/`, `runs/`, `wandb/`, checkpoints, videos, and Python caches out of git.
- Keep the small source assets in `go1_ppo/assets/` committed so MJCF files load without external paths.
- Add trained policies as release artifacts instead of committing checkpoint directories.
- Verify `python -c "import go1_ppo"` and MuJoCo model loading after fresh clones.

## License

This project follows the Apache-2.0 license used by the MuJoCo/MJX-derived
source files. Confirm that any added robot assets, textures, or trained policies
are redistributable before publishing them publicly.
