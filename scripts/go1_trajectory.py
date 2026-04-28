#!/usr/bin/env python3
"""Drive a trained Go1 PPO policy along a predefined trajectory (no keyboard needed)."""

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

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

from go1_ppo import registry          # noqa: E402
from go1_ppo.configs import locomotion_params  # noqa: E402

CONFIG_FNAME = "ppo_network_config.json"


# ---------------------------------------------------------------------------
# Trajectory primitives
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    x0: float
    x1: float
    y0: float
    y1: float
    yaw0: float
    yaw1: float
    duration: float

def quintic_blend(t, T, v0, vf):
    """Smooth transition from v0 to vf over time T."""
    if T <= 0:
        return vf
    tau = np.clip(t / T, 0.0, 1.0)
    s = 6*tau**5 - 15*tau**4 + 10*tau**3
    return v0 + (vf - v0) * s

def straight_line(speed=0.5):
    return [
        Segment(0.0, speed, 0, 0, 0, 0, 2.0),   # accelerate
        Segment(speed, speed, 0, 0, 0, 0, 1.0), # constant
        Segment(speed, 0.0, 0, 0, 0, 0, 2.0),   # decelerate
    ]

def circle(speed=0.4, radius=1.0):
    # Avoid divide-by-zero
    if abs(radius) < 1e-6:
        raise ValueError("Radius must be non-zero.")

    yaw_vel = speed / radius  # ω = v / R

    return [
        Segment(0.0, speed, 0, 0, 0.0, yaw_vel, 2.0),  
        Segment(speed, speed, 0, 0, yaw_vel, yaw_vel, 6.0),
        Segment(speed, 0.0, 0, 0, yaw_vel, 0.0, 2.0),
    ]

def from_json(path: str) -> List[Segment]:
    """
    Load a trajectory from a JSON file.

    Expected format::

        [
          {"x_vel": 0.5, "y_vel": 0.0, "yaw_vel": 0.0, "duration": 3.0},
          ...
        ]
    """
    data = json.loads(Path(path).read_text())
    return [Segment(**d) for d in data]


BUILT_IN_TRAJECTORIES = {
    "straight": straight_line,
    "circle": circle,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers  (identical to teleop script)
# ---------------------------------------------------------------------------

def _latest_step_dir(ckpt_root: Path) -> Path:
    step_dirs = [p for p in ckpt_root.iterdir()
                 if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(f"No numeric checkpoint dirs in: {ckpt_root}")
    return max(step_dirs, key=lambda p: int(p.name))


def _restore_params_orbax(step_dir: Path) -> Any:
    return ocp.PyTreeCheckpointer().restore(step_dir.as_posix())


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


def _find_obs_dim(params: Any) -> int:
    kernels: List[Tuple[int, int]] = []
    stack: List[Any] = [params]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif hasattr(node, "ndim") and hasattr(node, "shape"):
            try:
                if int(node.ndim) == 2:
                    kernels.append((int(node.shape[0]), int(node.shape[1])))
            except Exception:
                pass
    if not kernels:
        raise ValueError("Could not infer obs dim from policy params.")
    return min(kernels, key=lambda x: x[0])[0]


def _action_dim(env) -> int:
    sz = getattr(env, "action_size", None)
    if sz is None:
        raise ValueError("env has no action_size")
    return int(sum(int(v) for v in sz.values())) if isinstance(sz, Mapping) else int(sz)


def _ensure_network_config(step_dir: Path, env_name: str,
                            obs_dim: int, act_dim: int, force: bool):
    cfg_path = step_dir / CONFIG_FNAME
    if cfg_path.exists() and not force:
        return
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    cfg = {
        "observation_size": obs_dim,
        "action_size": act_dim,
        "normalize_observations": bool(
            getattr(ppo_params, "normalize_observations", True)),
        "network_factory_kwargs": _to_plain(ppo_params.network_factory),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------

class TrajectoryRunner:
    """Iterates through a list of Segments and returns the active command."""

    def __init__(self, segments: List[Segment], loop: bool = False):
        self.segments = segments
        self.loop = loop
        self._seg_idx = 0
        self._seg_elapsed = 0.0
        self.done = False

    def reset(self):
        self._seg_idx = 0
        self._seg_elapsed = 0.0
        self.done = False

    def step(self, dt: float) -> Tuple[float, float, float]:
        """Advance time by `dt` and return (x_vel, y_vel, yaw_vel)."""
        if self.done:
            return 0.0, 0.0, 0.0

        seg = self.segments[self._seg_idx]
        self._seg_elapsed += dt

        if self._seg_elapsed >= seg.duration:
            self._seg_elapsed = 0.0
            self._seg_idx += 1
            if self._seg_idx >= len(self.segments):
                if self.loop:
                    self._seg_idx = 0
                else:
                    self.done = True
                    return 0.0, 0.0, 0.0

        seg = self.segments[self._seg_idx]
        # return seg.x_vel, seg.y_vel, seg.yaw_vel
        t = self._seg_elapsed
        T = seg.duration

        x_vel = quintic_blend(t, T, seg.x0, seg.x1)
        y_vel = quintic_blend(t, T, seg.y0, seg.y1)
        yaw_vel = quintic_blend(t, T, seg.yaw0, seg.yaw1)

        return x_vel, y_vel, yaw_vel

    @property
    def progress(self) -> str:
        if self.done:
            return "done"
        seg = self.segments[self._seg_idx]
        pct = 100 * self._seg_elapsed / max(seg.duration, 1e-6)
        return f"seg {self._seg_idx + 1}/{len(self.segments)}  ({pct:.0f}%)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a Go1 PPO policy along a predefined trajectory.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory.")
    parser.add_argument("--env", type=str, default="",
                        help="Env name (inferred from run_dir if omitted).")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint step folder; default = latest.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force_rebuild_config", action="store_true")

    # Trajectory selection
    traj_group = parser.add_mutually_exclusive_group()
    traj_group.add_argument(
        "--trajectory",
        choices=list(BUILT_IN_TRAJECTORIES.keys()),
        default="straight",
        help="Built-in trajectory to run (default: straight).",
    )
    traj_group.add_argument(
        "--trajectory_json",
        type=str,
        default="",
        help="Path to a JSON file defining custom segments.",
    )

    # Trajectory tuning knobs
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Forward speed for built-in trajectories (m/s).")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Radius for the circular trajectory (m).")
    # parser.add_argument("--duration", type=float, default=5.0,
    #                     help="Duration of the straight-line segment (s).")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the trajectory indefinitely.")
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="Real-time factor (1.0 = real-time).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths / load env + policy
    # ------------------------------------------------------------------
    run_dir = Path(args.run_dir).expanduser().resolve()
    ckpt_root = run_dir / "checkpoints"
    env_name = args.env.strip() or run_dir.name.split("-")[0]

    step_dir = (
        (ckpt_root / args.ckpt).resolve() if args.ckpt
        else _latest_step_dir(ckpt_root)
    )

    env_cfg_path = run_dir / "env_config.json"
    if not env_cfg_path.exists():
        env_cfg_path = ckpt_root / "config.json"
    if not env_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing env config. Checked {run_dir/'env_config.json'} "
            f"and {ckpt_root/'config.json'}"
        )
    env_cfg = config_dict.ConfigDict(json.loads(env_cfg_path.read_text()))
    env = registry.load(env_name, config=env_cfg)

    params_tree = _restore_params_orbax(step_dir)
    obs_dim = _find_obs_dim(params_tree)
    act_dim = _action_dim(env)
    _ensure_network_config(step_dir, env_name, obs_dim, act_dim,
                           force=bool(args.force_rebuild_config))

    policy = ppo_checkpoint.load_policy(
        step_dir.as_posix(), deterministic=args.deterministic)
    jit_policy = jax.jit(policy)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # ------------------------------------------------------------------
    # Build trajectory
    # ------------------------------------------------------------------
    if args.trajectory_json:
        segments = from_json(args.trajectory_json)
        print(f"Loaded {len(segments)} segments from {args.trajectory_json}")
    else:
        fn = BUILT_IN_TRAJECTORIES[args.trajectory]
        # Pass relevant kwargs depending on which built-in was chosen
        if args.trajectory == "straight":
            segments = fn(speed=args.speed) # , duration=args.duration)
        elif args.trajectory == "circle":
            segments = fn(speed=args.speed, radius=args.radius)
        else:
            segments = fn()

    runner = TrajectoryRunner(segments, loop=args.loop)
    total_traj_time = sum(s.duration for s in segments)

    print(f"\n=== Go1 Trajectory Runner ===")
    print(f"  Trajectory : {args.trajectory_json or args.trajectory}")
    print(f"  Segments   : {len(segments)}")
    print(f"  Total time : {total_traj_time:.1f}s  (loop={args.loop})")
    print(f"  RTF        : {args.rtf}")
    print(f"  Checkpoint : {step_dir.name}\n")
    print("Press Ctrl-C to quit.\n")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    rtf = float(args.rtf)
    next_step_time = time.perf_counter()
    last_print = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            if runner.done and not args.loop:
                # Hold the final zero-command pose; keep viewer open
                time.sleep(0.05)
                viewer.sync()
                continue

            # Query trajectory for current command
            x_vel, y_vel, yaw_vel = runner.step(env.dt)

            # Inject command and step policy
            state.info["command"] = jp.array(
                [x_vel, y_vel, yaw_vel], dtype=jp.float32)
            rng, act_key = jax.random.split(rng)
            action, _ = jit_policy(state.obs, act_key)
            state = jit_step(state, action)

            # Sync MuJoCo viewer
            mj_data.qpos[:] = np.asarray(state.data.qpos)
            mj_data.qvel[:] = np.asarray(state.data.qvel)
            mj_data.ctrl[:] = np.asarray(state.data.ctrl)
            mj_data.time = float(state.data.time)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Console status
            now = time.perf_counter()
            if now - last_print > 0.25:
                print(
                    f"\rcmd=[{x_vel:+.2f}, {y_vel:+.2f}, {yaw_vel:+.2f}]  "
                    f"reward={float(state.reward):+.3f}  "
                    f"t={float(state.data.time):.2f}s  "
                    f"{runner.progress}  rtf={rtf:.2f}   ",
                    end="", flush=True,
                )
                last_print = now

            # Real-time pacing
            next_step_time += env.dt / rtf
            sleep_dt = next_step_time - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_step_time = time.perf_counter()

    print("\nTrajectory runner exited.")


if __name__ == "__main__":
    main()