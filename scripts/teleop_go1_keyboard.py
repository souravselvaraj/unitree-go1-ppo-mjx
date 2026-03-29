#!/usr/bin/env python3
"""Load a trained Go1 PPO policy and drive it live with the keyboard."""

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Tuple

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
import mujoco.viewer
import numpy as np
from orbax import checkpoint as ocp

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT.parent))

from go1_ppo import registry  # noqa: E402
from go1_ppo.configs import locomotion_params  # noqa: E402

CONFIG_FNAME = "ppo_network_config.json"


class RawTerminal:
  """Context manager to read single characters from the terminal."""

  def __enter__(self):
    import termios
    import tty

    self.fd = sys.stdin.fileno()
    self.old = termios.tcgetattr(self.fd)
    tty.setcbreak(self.fd)
    return self

  def __exit__(self, exc_type, exc, tb):
    import termios

    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

  @staticmethod
  def get_key_nonblocking():
    import select

    if select.select([sys.stdin], [], [], 0.0)[0]:
      return sys.stdin.read(1)
    return None


def _latest_step_dir(ckpt_root: Path) -> Path:
  step_dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()]
  if not step_dirs:
    raise FileNotFoundError(f"No numeric checkpoint step dirs found in: {ckpt_root}")
  step_dirs.sort(key=lambda p: int(p.name))
  return step_dirs[-1]


def _restore_params_orbax(step_dir: Path) -> Any:
  ckptr = ocp.PyTreeCheckpointer()
  return ckptr.restore(step_dir.as_posix())


def _to_plain(x: Any) -> Any:
  try:
    if isinstance(x, (jax.Array, jp.ndarray)):
      x = jp.asarray(x)
      return float(x) if x.ndim == 0 else np.array(x).tolist()
  except Exception:
    pass

  if hasattr(x, "to_dict"):
    return _to_plain(x.to_dict())
  if isinstance(x, Mapping):
    return {str(k): _to_plain(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_to_plain(v) for v in x]
  return x


def _find_obs_dim_from_params(params: Any) -> int:
  kernels: List[Tuple[int, int]] = []
  stack: List[Any] = [params]

  while stack:
    node = stack.pop()
    if isinstance(node, Mapping):
      stack.extend(list(node.values()))
      continue
    if isinstance(node, (list, tuple)):
      stack.extend(list(node))
      continue

    if hasattr(node, "ndim") and hasattr(node, "shape"):
      try:
        if int(node.ndim) == 2:
          kernels.append((int(node.shape[0]), int(node.shape[1])))
      except Exception:
        pass

  if not kernels:
    raise ValueError("Could not infer observation size from policy params.")
  kernels.sort(key=lambda x: x[0])
  return kernels[0][0]


def _action_dim_from_env(env) -> int:
  action_size = getattr(env, "action_size", None)
  if action_size is None:
    raise ValueError("Environment has no action_size.")
  if isinstance(action_size, Mapping):
    return int(sum(int(v) for v in action_size.values()))
  return int(action_size)


def _ensure_step_network_config(
    step_dir: Path, env_name: str, obs_dim: int, act_dim: int, force: bool
):
  cfg_path = step_dir / CONFIG_FNAME
  if cfg_path.exists() and not force:
    return

  ppo_params = locomotion_params.brax_ppo_config(env_name)
  cfg = {
      "observation_size": int(obs_dim),
      "action_size": int(act_dim),
      "normalize_observations": bool(
          getattr(ppo_params, "normalize_observations", True)
      ),
      "network_factory_kwargs": _to_plain(ppo_params.network_factory),
  }
  cfg_path.write_text(json.dumps(cfg, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_dir", type=str, required=True)
  parser.add_argument(
      "--env",
      type=str,
      default="",
      help="Optional override; else infer from run directory name.",
  )
  parser.add_argument(
      "--ckpt",
      type=str,
      default="",
      help="Optional numeric checkpoint step folder; default is latest.",
  )
  parser.add_argument("--deterministic", action="store_true")
  parser.add_argument("--force_rebuild_config", action="store_true")
  parser.add_argument("--x_vel", type=float, default=0.5)
  parser.add_argument("--y_vel", type=float, default=0.0)
  parser.add_argument("--yaw_vel", type=float, default=0.0)
  parser.add_argument("--dv", type=float, default=0.05)
  parser.add_argument("--dw", type=float, default=0.10)
  parser.add_argument("--rtf", type=float, default=1.0)
  args = parser.parse_args()

  run_dir = Path(args.run_dir).expanduser().resolve()
  ckpt_root = (run_dir / "checkpoints").resolve()
  env_name = args.env.strip() if args.env.strip() else run_dir.name.split("-")[0]

  if args.ckpt:
    step_dir = (ckpt_root / args.ckpt).resolve()
    if not step_dir.exists():
      raise FileNotFoundError(f"Checkpoint step dir not found: {step_dir}")
  else:
    step_dir = _latest_step_dir(ckpt_root)

  env_cfg_path = run_dir / "env_config.json"
  if not env_cfg_path.exists():
    env_cfg_path = ckpt_root / "config.json"
  if not env_cfg_path.exists():
    raise FileNotFoundError(
        "Missing environment config. Checked:\n"
        f"  - {run_dir / 'env_config.json'}\n"
        f"  - {ckpt_root / 'config.json'}"
    )
  env_cfg = config_dict.ConfigDict(json.loads(env_cfg_path.read_text()))
  env = registry.load(env_name, config=env_cfg)

  params_tree = _restore_params_orbax(step_dir)
  obs_dim = _find_obs_dim_from_params(params_tree)
  act_dim = _action_dim_from_env(env)
  _ensure_step_network_config(
      step_dir,
      env_name,
      obs_dim,
      act_dim,
      force=bool(args.force_rebuild_config),
  )

  policy = ppo_checkpoint.load_policy(
      step_dir.as_posix(), deterministic=args.deterministic
  )
  jit_policy = jax.jit(policy)
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  mj_model = env.mj_model
  mj_data = mujoco.MjData(mj_model)

  x_vel = float(args.x_vel)
  y_vel = float(args.y_vel)
  yaw_vel = float(args.yaw_vel)
  dv = float(args.dv)
  dw = float(args.dw)
  rtf = float(args.rtf)
  paused = False

  print("\n=== Go1 Live Keyboard Teleop ===")
  print("Focus the terminal window for controls:")
  print("  w/s: +/- x_vel")
  print("  a/d: +/- y_vel")
  print("  q/e: +/- yaw_vel")
  print("  0: zero command")
  print("  [ / ]: slower/faster")
  print("  space: pause")
  print("  r: reset")
  print("  x: quit\n")

  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)

  with RawTerminal(), mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    last_print = 0.0
    next_step_time = time.perf_counter()

    while viewer.is_running():
      key = RawTerminal.get_key_nonblocking()
      if key:
        key = key.lower()
        if key == "x":
          break
        elif key == " ":
          paused = not paused
        elif key == "r":
          rng, reset_key = jax.random.split(rng)
          state = jit_reset(reset_key)
        elif key == "w":
          x_vel += dv
        elif key == "s":
          x_vel -= dv
        elif key == "a":
          y_vel += dv
        elif key == "d":
          y_vel -= dv
        elif key == "q":
          yaw_vel += dw
        elif key == "e":
          yaw_vel -= dw
        elif key == "0":
          x_vel, y_vel, yaw_vel = 0.0, 0.0, 0.0
        elif key == "[":
          rtf = max(0.1, rtf * 0.8)
        elif key == "]":
          rtf = min(10.0, rtf * 1.25)

      if paused:
        time.sleep(0.01)
        continue

      state.info["command"] = jp.array([x_vel, y_vel, yaw_vel], dtype=jp.float32)
      rng, act_key = jax.random.split(rng)
      action, _ = jit_policy(state.obs, act_key)
      state = jit_step(state, action)

      mj_data.qpos[:] = np.asarray(state.data.qpos)
      mj_data.qvel[:] = np.asarray(state.data.qvel)
      mj_data.ctrl[:] = np.asarray(state.data.ctrl)
      mj_data.time = float(state.data.time)
      mujoco.mj_forward(mj_model, mj_data)
      viewer.sync()

      now = time.perf_counter()
      if now - last_print > 0.25:
        print(
            f"\rcommand = [{x_vel:+.2f}, {y_vel:+.2f}, {yaw_vel:+.2f}]   "
            f"reward = {float(state.reward):+.3f}   "
            f"time = {float(state.data.time):.2f}s   "
            f"rtf = {rtf:.2f}",
            end="",
            flush=True,
        )
        last_print = now

      next_step_time += env.dt / rtf
      sleep_dt = next_step_time - time.perf_counter()
      if sleep_dt > 0:
        time.sleep(sleep_dt)
      else:
        next_step_time = time.perf_counter()

  print("\nExited teleop.")


if __name__ == "__main__":
  main()
