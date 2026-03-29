# go1_ppo

Standalone Go1 PPO workspace for MuJoCo.

## Setup

```bash
cd /home/sourav/skadi/go1_ppo
uv sync
source /home/sourav/skadi/go1_ppo/.venv/bin/activate
```

## Train

Quick test:

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True --num_timesteps=10000 --num_envs=256 --num_eval_envs=32 --num_evals=2
```

Default flat-terrain training:

```bash
python /home/sourav/skadi/go1_ppo/training/train_jax_ppo.py --env_name=Go1JoystickFlatTerrain --use_tb=True
```

## TensorBoard

```bash
tensorboard --logdir /home/sourav/skadi/go1_ppo/logs --port 6006
```

Open `http://localhost:6006`

## Teleop

```bash
python /home/sourav/skadi/go1_ppo/scripts/teleop_go1_keyboard.py --run_dir /home/sourav/skadi/go1_ppo/logs/Go1JoystickFlatTerrain-YYYYMMDD-HHMMSS --deterministic
```

Controls:

- `w/s` forward/backward
- `a/d` lateral
- `q/e` yaw
- `0` zero command
- `space` pause
- `r` reset
- `x` quit
