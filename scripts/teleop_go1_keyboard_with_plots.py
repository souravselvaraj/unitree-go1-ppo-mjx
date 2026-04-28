#!/usr/bin/env python3
"""Load a trained Go1 PPO policy and drive it live with the keyboard.
Generates plots of commanded vs simulated velocities and joint angles.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
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

# Try to import matplotlib - will generate plots if available
try:
  import matplotlib
  matplotlib.use('Agg')  # Non-interactive backend
  import matplotlib.pyplot as plt
  MATPLOTLIB_AVAILABLE = True
except ImportError:
  MATPLOTLIB_AVAILABLE = False
  print("Warning: matplotlib not available, plots will not be generated")


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


def _get_joint_names(mj_model) -> List[str]:
  """Extract joint names from MuJoCo model."""
  joint_names = []
  for i in range(mj_model.njnt):
    if mj_model.jnt_type[i] != 0:  # Not free joint
      name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if name:
        joint_names.append(name)
  return joint_names


def _compute_base_velocity(mj_data) -> np.ndarray:
  """Compute base linear and angular velocity from MuJoCo data.
  
  Returns:
    np.ndarray: [v_x, v_y, omega_z] - forward, lateral velocities and yaw rate
  """
  # Get root quaternion
  quat = mj_data.qpos[3:7]  # [qw, qx, qy, qz]
  
  # Convert quaternion to rotation matrix
  # MuJoCo quaternion is [w, x, y, z]
  qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
  
  # Rotation matrix from body to world
  R = np.array([
    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
  ])
  
  # World frame velocity
  world_vel = mj_data.qvel[:3]
  world_ang = mj_data.qvel[3:6]
  
  # Transform to body frame
  body_vel = R.T @ world_vel
  body_ang = R.T @ world_ang
  
  # Return [forward, lateral, yaw_rate]
  return np.array([body_vel[0], body_vel[1], body_ang[2]])


def _generate_plots(time_history, cmd_history, vel_history, joint_history, joint_names, output_dir):
  """Generate and save plots."""
  if not MATPLOTLIB_AVAILABLE:
    print("Skipping plot generation (matplotlib not available)")
    return

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  time_arr = np.array(time_history)
  cmd_arr = np.array(cmd_history)
  vel_arr = np.array(vel_history)
  joint_arr = np.array(joint_history)
  
  # Plot 1: Commanded vs Simulated Velocities
  fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
  labels = ['Forward (m/s)', 'Lateral (m/s)', 'Yaw Rate (rad/s)']
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
  
  for i, (ax, label, color) in enumerate(zip(axes1, labels, colors)):
    ax.plot(time_arr, cmd_arr[:, i], label='Commanded', color=color, linewidth=1.5, alpha=0.8)
    ax.plot(time_arr, vel_arr[:, i], label='Simulated', color=color, linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_ylabel(label)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_arr[0], time_arr[-1])
  
  axes1[-1].set_xlabel('Time (s)')
  fig1.suptitle('Commanded vs Simulated Velocities', fontsize=14, fontweight='bold')
  plt.tight_layout()
  fig1.savefig(output_dir / 'velocity_commands.png', dpi=150, bbox_inches='tight')
  plt.close(fig1)
  print(f"Saved: {output_dir / 'velocity_commands.png'}")
  
  # Plot 2: Joint Angles over Time
  # Group joints by leg
  leg_joints = {
    'FR': ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
    'FL': ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
    'RR': ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
    'RL': ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
  }
  
  joint_type_labels = {
    'hip_joint': 'Hip (abduction)',
    'thigh_joint': 'Thigh',
    'calf_joint': 'Knee',
  }
  
  # Plot each joint type across all legs
  for joint_type in ['hip_joint', 'thigh_joint', 'calf_joint']:
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle(f'{joint_type_labels.get(joint_type, joint_type)} Joint Angles', 
                  fontsize=14, fontweight='bold')
    
    for leg_idx, (leg, joints) in enumerate(leg_joints.items()):
      ax = axes2[leg_idx]
      for jnt in joints:
        if joint_type in jnt.lower():
          # Find the column index for this joint
          jnt_idx = None
          for idx, name in enumerate(joint_names):
            if name == jnt:
              jnt_idx = idx
              break
          
          if jnt_idx is not None and jnt_idx < joint_arr.shape[1]:
            ax.plot(time_arr, np.degrees(joint_arr[:, jnt_idx]), 
                   label=jnt, linewidth=1.2, alpha=0.8)
    
    for ax, leg in zip(axes2, leg_joints.keys()):
      ax.set_ylabel(f'{leg} (deg)')
      ax.legend(loc='upper right', fontsize=8)
      ax.grid(True, alpha=0.3)
    
    axes2[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    filename = f'joint_{joint_type.replace("_", "_")}.png'
    fig2.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / filename}")
  
  # Plot 3: All joints combined heatmap
  if joint_arr.shape[1] > 0:
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    # Convert to degrees for better visualization
    joint_deg = np.degrees(joint_arr)
    
    # Create heatmap
    im = ax3.imshow(joint_deg.T, aspect='auto', cmap='RdBu_r', 
                    extent=[time_arr[0], time_arr[-1], 0, joint_arr.shape[1]])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Joint Index')
    ax3.set_title('Joint Angles Heatmap (degrees)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Angle (deg)')
    
    # Add joint name labels on y-axis
    tick_positions = np.arange(len(joint_names)) + 0.5
    ax3.set_yticks(tick_positions[::3])  # Show every 3rd label to avoid crowding
    ax3.set_yticklabels(joint_names[::3], fontsize=8)
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'joint_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir / 'joint_heatmap.png'}")
  
  print(f"\nAll plots saved to: {output_dir}")


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
  parser.add_argument("--plot_dir", type=str, default="", 
                      help="Directory to save plots (default: run_dir/plots)")
  parser.add_argument("--max_history", type=int, default=10000,
                      help="Maximum number of timesteps to record")
  args = parser.parse_args()

  run_dir = Path(args.run_dir).expanduser().resolve()
  ckpt_root = (run_dir / "checkpoints").resolve()
  
  # Plot output directory
  plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "plots"
  
  # Environment selection
  available_envs = list(registry.ALL_ENVS)
  if args.env.strip():
    env_name = args.env.strip()
    if env_name not in available_envs:
      raise ValueError(f"Environment '{env_name}' not found. Available: {available_envs}")
  else:
    print("\n=== Available Environments ===")
    for i, env in enumerate(available_envs, 1):
      print(f"  {i}. {env}")
    
    while True:
      try:
        choice = input(f"\nSelect environment (1-{len(available_envs)}): ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(available_envs):
          env_name = available_envs[idx]
          print(f"Selected: {env_name}\n")
          break
        else:
          print(f"Invalid choice. Enter 1-{len(available_envs)}")
      except ValueError:
        print(f"Invalid input. Enter a number 1-{len(available_envs)}")

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

  # Get joint names for plotting
  joint_names = _get_joint_names(mj_model)
  print(f"Detected {len(joint_names)} joints: {joint_names}")

  x_vel = float(args.x_vel)
  y_vel = float(args.y_vel)
  yaw_vel = float(args.yaw_vel)
  dv = float(args.dv)
  dw = float(args.dw)
  rtf = float(args.rtf)
  paused = False

  # Data recording buffers
  time_history = []
  cmd_history = []  # Commanded velocities
  vel_history = []  # Simulated velocities
  joint_history = []  # Joint angles

  print("\n=== Go1 Live Keyboard Teleop with Plotting ===")
  print("Focus the terminal window for controls:")
  print("  w/s: +/- x_vel")
  print("  a/d: +/- y_vel")
  print("  q/e: +/- yaw_vel")
  print("  0: zero command")
  print("  [ / ]: slower/faster")
  print("  space: pause")
  print("  r: reset")
  print("  p: generate plots now")
  print("  x: quit and generate plots\n")

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
        elif key == "p":
          # Generate plots on demand
          print("\nGenerating plots...")
          _generate_plots(time_history, cmd_history, vel_history, joint_history, joint_names, plot_dir)
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

      # Record data
      if len(time_history) < args.max_history:
        # Time
        time_history.append(float(mj_data.time))
        
        # Commanded velocity
        cmd_history.append([x_vel, y_vel, yaw_vel])
        
        # Simulated velocity (computed from MuJoCo state)
        sim_vel = _compute_base_velocity(mj_data)
        vel_history.append(sim_vel.tolist())
        
        # Joint angles (skip free joint qpos[:6] = [x, y, z, qw, qx, qy, qz])
        joint_angles = np.asarray(mj_data.qpos[7:]).tolist()
        joint_history.append(joint_angles)

      now = time.perf_counter()
      if now - last_print > 0.25:
        print(
            f"\rcommand = [{x_vel:+.2f}, {y_vel:+.2f}, {yaw_vel:+.2f}]   "
            f"reward = {float(state.reward):+.3f}   "
            f"time = {float(state.data.time):.2f}s   "
            f"rtf = {rtf:.2f}   "
            f"records = {len(time_history)}",
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

  print("\n\nExited teleop.")
  
  # Generate plots on exit
  if time_history:
    print(f"\nGenerating plots from {len(time_history)} recorded timesteps...")
    _generate_plots(time_history, cmd_history, vel_history, joint_history, joint_names, plot_dir)
  else:
    print("No data recorded, skipping plot generation.")


if __name__ == "__main__":
  main()