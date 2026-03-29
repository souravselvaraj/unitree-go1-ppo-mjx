# go1_ppo

Standalone PPO training workspace for Unitree Go1 locomotion in MuJoCo.

## Overview

This project contains a self-contained Go1 locomotion setup built around:

- Go1 flat-terrain and rough-terrain joystick tasks
- Brax PPO training on top of MJX / MuJoCo
- local MuJoCo scenes and mesh assets
- checkpointed training runs
- keyboard teleoperation for trained policies

The local Go1 reward logic and PPO config are aligned with the main `mujoco_playground` tree for the Go1 task paths, while the package remains standalone.

## Layout

- [`registry.py`](/home/sourav/skadi/go1_ppo/registry.py): local environment registry
- [`mjx_env.py`](/home/sourav/skadi/go1_ppo/mjx_env.py): MJX env base utilities, rendering, state container
- [`wrapper.py`](/home/sourav/skadi/go1_ppo/wrapper.py): Brax wrappers, auto-reset, domain-randomization wrapper
- [`locomotion.py`](/home/sourav/skadi/go1_ppo/locomotion.py): local shim for locomotion config lookup
- [`envs/base.py`](/home/sourav/skadi/go1_ppo/envs/base.py): Go1 base env and asset loading
- [`envs/joystick.py`](/home/sourav/skadi/go1_ppo/envs/joystick.py): Go1 locomotion task definition
- [`envs/randomize.py`](/home/sourav/skadi/go1_ppo/envs/randomize.py): domain randomization
- [`configs/locomotion_params.py`](/home/sourav/skadi/go1_ppo/configs/locomotion_params.py): PPO hyperparameters
- [`training/train_jax_ppo.py`](/home/sourav/skadi/go1_ppo/training/train_jax_ppo.py): main training entrypoint
- [`scripts/train_go1_ppo.py`](/home/sourav/skadi/go1_ppo/scripts/train_go1_ppo.py): simple standalone trainer
- [`scripts/teleop_go1_keyboard.py`](/home/sourav/skadi/go1_ppo/scripts/teleop_go1_keyboard.py): live keyboard control of a trained policy
- [`scene/`](/home/sourav/skadi/go1_ppo/scene): MuJoCo XML scenes
- [`assets/`](/home/sourav/skadi/go1_ppo/assets): meshes and terrain textures
- [`logs/`](/home/sourav/skadi/go1_ppo/logs): training logs and checkpoints

## Supported Environments

- `Go1JoystickFlatTerrain`
- `Go1JoystickRoughTerrain`

## Environment Setup

Python version is pinned in [`.python-version`](/home/sourav/skadi/go1_ppo/.python-version).

Create or sync the venv:

```bash
cd /home/sourav/skadi/go1_ppo
uv sync
source /home/sourav/skadi/go1_ppo/.venv/bin/activate
```

## Training

### Quick Smoke Test

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True --num_timesteps=10000 --num_envs=256 --num_eval_envs=32 --num_evals=2
```

### Standard Flat-Terrain Training

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True
```

### Larger Training Run

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True --num_timesteps=10000000 --num_envs=8192 --num_eval_envs=1024 --num_evals=100
```

If you use `--num_envs=16384`, make the PPO batch shape valid too:

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True --num_timesteps=10000000 --num_envs=16384 --num_eval_envs=4096 --num_evals=100 --batch_size=512
```

### Simpler Standalone Trainer

```bash
python /home/sourav/skadi/go1_ppo/scripts/train_go1_ppo.py --env Go1JoystickFlatTerrain --num_timesteps 1000000
```

## TensorBoard

Train with TensorBoard logging enabled:

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True
```

In another terminal:

```bash
source /home/sourav/skadi/go1_ppo/.venv/bin/activate
tensorboard --logdir /home/sourav/skadi/go1_ppo/logs --port 6006
```

Then open:

```text
http://localhost:6006
```

## Keyboard Teleop

Run a trained checkpoint in the MuJoCo viewer:

```bash
python /home/sourav/skadi/go1_ppo/scripts/teleop_go1_keyboard.py --run_dir /home/sourav/skadi/go1_ppo/logs/Go1JoystickFlatTerrain-YYYYMMDD-HHMMSS --deterministic
```

Use a specific checkpoint step:

```bash
python /home/sourav/skadi/go1_ppo/scripts/teleop_go1_keyboard.py --run_dir /home/sourav/skadi/go1_ppo/logs/Go1JoystickFlatTerrain-YYYYMMDD-HHMMSS --ckpt 000031129600 --deterministic
```

Controls:

- `w/s`: forward/backward
- `a/d`: lateral velocity
- `q/e`: yaw rate
- `0`: zero command
- `space`: pause
- `r`: reset
- `x`: quit
- `[` and `]`: slower/faster sim

## Domain Randomization

Domain randomization perturbs simulation properties during training, such as:

- floor friction
- joint friction loss
- armature
- body mass
- torso center of mass
- default joint pose

This is implemented in [`envs/randomize.py`](/home/sourav/skadi/go1_ppo/envs/randomize.py).

For a baseline flat-terrain run, it is usually better to train without domain randomization first. Enable it later for robustness experiments.

## Outputs

Training runs write under [`logs/`](/home/sourav/skadi/go1_ppo/logs):

- `checkpoints/`
- config snapshots
- TensorBoard event files when `--use_tb=True`
- rollout videos from the richer trainer

The simple script may also create run directories under `runs/`.

## Notes

- `warp` is optional. The JAX path is sufficient for normal PPO training.
- If TensorBoard fails with `pkg_resources` issues, install a compatible `setuptools` version in the venv.
- For teleop, use a checkpoint from a run with good eval reward, not necessarily the last checkpoint.
