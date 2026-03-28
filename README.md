# go1_ppo

Standalone PPO training workspace for Unitree Go1 locomotion 
T
## What is in this repo

- [registry.py](/home/sourav/skadi/go1_ppo/registry.py): environment registration and config lookup
- [mjx_env.py](/home/sourav/skadi/go1_ppo/mjx_env.py): shared MJX environment base class and rendering helpers
- [wrapper.py](/home/sourav/skadi/go1_ppo/wrapper.py): Brax-compatible training wrappers, autoreset, and domain-randomization vmap wrapper
- [envs/base.py](/home/sourav/skadi/go1_ppo/envs/base.py): Go1 base environment, MuJoCo model loading, actuator setup
- [envs/joystick.py](/home/sourav/skadi/go1_ppo/envs/joystick.py): joystick locomotion task definition
- [envs/randomize.py](/home/sourav/skadi/go1_ppo/envs/randomize.py): domain-randomization utilities
- [configs/locomotion_params.py](/home/sourav/skadi/go1_ppo/configs/locomotion_params.py): PPO hyperparameters
- [scripts/train_go1_ppo.py](/home/sourav/skadi/go1_ppo/scripts/train_go1_ppo.py): practical standalone trainer that writes run artifacts under `runs/`
- [training/train_jax_ppo.py](/home/sourav/skadi/go1_ppo/training/train_jax_ppo.py): flag-driven training and playback script that can render rollout videos
- [scene/](/home/sourav/skadi/go1_ppo/scene): MuJoCo XML scenes
- [assets/](/home/sourav/skadi/go1_ppo/assets): meshes and terrain textures

## Supported environments

The local registry currently exposes:

- `Go1JoystickFlatTerrain`
- `Go1JoystickRoughTerrain`

The default environment is `Go1JoystickFlatTerrain`.

## Requirements

- Python `>=3.12,<3.13`
- `uv` recommended for dependency management
- MuJoCo-compatible Linux environment
- CUDA-capable setup if you want GPU-backed JAX

Dependencies are declared in [pyproject.toml](/home/sourav/skadi/go1_ppo/pyproject.toml). Key packages include:

- `jax[cuda12]`
- `brax>=0.12.5`
- `mujoco>=3.4.dev`
- `mujoco-mjx>=3.4.dev`
- `mediapy`
- `orbax-checkpoint`

## Setup

### 1. Create or sync the environment

```bash
uv sync
```



### 2. Use the repo from its parent directory on `PYTHONPATH` when needed

Some direct module invocations may need the parent of this repo on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/parent/of/go1_ppo
```

The included scripts already patch `sys.path` for their own execution, so this is mainly useful for ad hoc Python commands.

## Quick start

### Standalone training run

```bash
uv run python scripts/train_go1_ppo.py --env Go1JoystickFlatTerrain
```

This creates a timestamped run directory under `runs/`.

### Override the total number of timesteps

```bash
uv run python scripts/train_go1_ppo.py \
  --env Go1JoystickFlatTerrain \
  --num_timesteps 1000000
```

### Disable domain randomization

```bash
uv run python scripts/train_go1_ppo.py \
  --env Go1JoystickFlatTerrain \
  --no_domain_rand
```

### Choose a custom run directory

```bash
uv run python scripts/train_go1_ppo.py \
  --env Go1JoystickFlatTerrain \
  --run_dir ./runs/debug-flat
```

## Training entry points

There are two main ways to run training in this workspace.

### 1. `scripts/train_go1_ppo.py`

Use [scripts/train_go1_ppo.py](/home/sourav/skadi/go1_ppo/scripts/train_go1_ppo.py) for the simplest standalone workflow.

What it does:

- loads the selected environment from the local registry
- fetches the tuned Brax PPO config from `configs/locomotion_params.py`
- optionally enables domain randomization
- writes run metadata and metrics to disk
- saves Orbax checkpoints using absolute paths

CLI arguments:

- `--env`: environment name, default `Go1JoystickFlatTerrain`
- `--run_dir`: optional output directory
- `--seed`: random seed, default `0`
- `--num_timesteps`: override PPO config if greater than `0`
- `--no_domain_rand`: disable domain randomization

Typical output:

```text
runs/
  Go1JoystickFlatTerrain-YYYYMMDD-HHMMSS/
    env_config.json
    ppo_config.json
    progress.csv
    final_metrics.json
    checkpoints/
```

### 2. `training/train_jax_ppo.py`

Use [training/train_jax_ppo.py](/home/sourav/skadi/go1_ppo/training/train_jax_ppo.py) if you want the more feature-rich training script with Abseil flags, evaluation rollouts, and video export.

Examples:

```bash
uv run python -m training.train_jax_ppo \
  --env_name=Go1JoystickFlatTerrain \
  --num_timesteps=1000000
```

Play only with an existing checkpoint:

```bash
uv run python -m training.train_jax_ppo \
  --env_name=Go1JoystickFlatTerrain \
  --play_only=true \
  --load_checkpoint_path=/absolute/path/to/checkpoint
```

Record evaluation videos after training:

```bash
uv run python -m training.train_jax_ppo \
  --env_name=Go1JoystickFlatTerrain \
  --num_videos=1
```

Important flags in this script include:

- `--env_name`
- `--impl`
- `--load_checkpoint_path`
- `--play_only`
- `--domain_randomization`
- `--seed`
- `--num_timesteps`
- `--num_videos`
- `--num_evals`
- `--episode_length`
- `--num_envs`
- `--num_eval_envs`
- `--batch_size`
- `--run_evals`

This script is the one that writes rollout videos like `rollout0.mp4`.

## PPO configuration

The tuned Go1 PPO defaults live in [configs/locomotion_params.py](/home/sourav/skadi/go1_ppo/configs/locomotion_params.py).

For `Go1JoystickFlatTerrain` and `Go1JoystickRoughTerrain`, the current defaults are:

- `num_timesteps = 200_000_000`
- `num_evals = 10`
- `num_envs = 8192`
- `unroll_length = 20`
- `num_minibatches = 32`
- `num_updates_per_batch = 4`
- `learning_rate = 3e-4`
- `entropy_cost = 1e-2`
- policy network: `(512, 256, 128)`
- value network: `(512, 256, 128)`
- policy observation key: `state`
- value observation key: `privileged_state`

Environment defaults from [envs/joystick.py](/home/sourav/skadi/go1_ppo/envs/joystick.py) include:

- control timestep `ctrl_dt = 0.02`
- simulation timestep `sim_dt = 0.004`
- episode length `1000`
- PD gains `Kp = 35.0`, `Kd = 0.5`
- action scale `0.5`

## Checkpoints and logs

The standalone trainer writes:

- `env_config.json`: serialized environment config used for the run
- `ppo_config.json`: serialized PPO settings
- `progress.csv`: periodic reward metrics and wall-clock time
- `final_metrics.json`: final summary metrics
- `checkpoints/`: Orbax checkpoint directory

The repo may also contain:

- `logs/`: prior training logs and checkpoint snapshots
- `runs/`: standalone script outputs

## Rendering and videos

Video export uses MuJoCo rendering plus `mediapy`.

The shared rendering code is in [mjx_env.py](/home/sourav/skadi/go1_ppo/mjx_env.py). The playback script:

1. rolls out the policy in the evaluation environment
2. subsamples frames with `render_every = 2`
3. renders the trajectory
4. writes `rollout{i}.mp4`

### Recent fix for video-render crashes

The render path was updated to avoid buffering the full video in memory before encoding.

The current behavior:

- streams frames directly to `mediapy` during export
- converts JAX arrays to NumPy before copying them into `mujoco.MjData`
- keeps the old `render()` API available for callers that still want a materialized frame list

Relevant code:

- [mjx_env.py](/home/sourav/skadi/go1_ppo/mjx_env.py)
- [training/train_jax_ppo.py](/home/sourav/skadi/go1_ppo/training/train_jax_ppo.py)

If you previously saw crashes or abrupt exits while writing `rollout0.mp4`, this fix is the first thing to try.

## Headless and Linux notes

The training playback script sets:

```text
MUJOCO_GL=egl
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

That makes headless rendering more reliable on Linux systems with EGL support.

If you run ad hoc rendering code outside the provided scripts, it can help to export:

```bash
export MUJOCO_GL=egl
```

## Domain randomization

Domain randomization is wired through the local registry and wrapper stack:

- [registry.py](/home/sourav/skadi/go1_ppo/registry.py)
- [wrapper.py](/home/sourav/skadi/go1_ppo/wrapper.py)
- [envs/randomize.py](/home/sourav/skadi/go1_ppo/envs/randomize.py)

In the standalone trainer:

- enabled by default
- disabled with `--no_domain_rand`

In the Abseil training script:

- controlled with `--domain_randomization`

## How the environment is built

The Go1 environment loads XML and mesh assets from the local repo, then constructs:

- a MuJoCo `MjModel` for rendering and host-side physics state transfer
- an MJX model for JAX-based stepping and training

The base class in [envs/base.py](/home/sourav/skadi/go1_ppo/envs/base.py):

- loads XML from `scene/`
- loads meshes and textures from `assets/`
- applies PD gains to actuators
- increases offscreen framebuffer size for higher-resolution rendering

## Troubleshooting

### `ModuleNotFoundError: No module named 'go1_ppo'`

Use the provided scripts directly or ensure the parent directory of this repo is on `PYTHONPATH`.

### JAX CUDA plugin errors

If JAX reports CUDA initialization failures, verify:

- the machine actually has a visible CUDA device
- the installed CUDA runtime matches the JAX wheel
- you are not forcing GPU execution in a CPU-only environment

### Rendering or playback crashes

Try the following:

- use the updated code in this repo with the streaming render fix
- set `MUJOCO_GL=egl`
- reduce `--num_videos`
- reduce `--episode_length`
- test on flat terrain first

### Orbax checkpoint path issues

The standalone trainer already resolves checkpoint paths to absolute paths because Orbax can fail on relative checkpoint locations.

## Development notes

Basic syntax verification:

```bash
python -m py_compile mjx_env.py training/train_jax_ppo.py scripts/train_go1_ppo.py
```

If you install dev dependencies, you can also run your preferred formatter and linter on top of that.

## License and attribution

This workspace is derived from the MuJoCo Playground and Brax ecosystem and retains upstream copyright headers in the imported or adapted source files.
