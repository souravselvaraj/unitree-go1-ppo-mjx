#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import jax
from jax import numpy as jp

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from go1_ppo import registry, wrapper
from go1_ppo.configs import locomotion_params


def _jsonable(x: Any) -> Any:
    """Best-effort conversion so configs/metrics can be json-dumped."""
    try:
        import numpy as np
        if isinstance(x, (jax.Array, jp.ndarray)):
            x = jp.asarray(x)
            return float(x) if x.ndim == 0 else x.tolist()
        if isinstance(x, np.ndarray):
            return float(x) if x.ndim == 0 else x.tolist()
    except Exception:
        pass

    if hasattr(x, "to_dict"):
        return _jsonable(x.to_dict())

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]

    return x


def _set_xla_flags():
    # Optional speedup on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
        xla_flags = (xla_flags + " --xla_gpu_triton_gemm_any=True").strip()
    os.environ["XLA_FLAGS"] = xla_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", dest="env_name", default="Go1JoystickFlatTerrain")
    parser.add_argument("--run_dir", type=str, default="", help="Optional. If empty, auto-create in ./runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=-1, help="Override PPO num_timesteps if > 0")
    parser.add_argument("--no_domain_rand", action="store_true", help="Disable env domain randomization")
    args = parser.parse_args()

    _set_xla_flags()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"{args.env_name}-{ts}"
    # >>> CRITICAL FIX: Orbax requires ABSOLUTE checkpoint paths
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = (run_dir / "checkpoints").expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_name = args.env_name
    env_cfg = registry.get_default_config(env_name)

    # Load training env
    env = registry.load(env_name, config=env_cfg)

    # PPO config from the standalone go1_ppo package
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    if args.num_timesteps and args.num_timesteps > 0:
        ppo_params["num_timesteps"] = int(args.num_timesteps)

    # Save configs
    (run_dir / "env_config.json").write_text(json.dumps(_jsonable(env_cfg), indent=2))
    (run_dir / "ppo_config.json").write_text(json.dumps(_jsonable(ppo_params), indent=2))

    # Domain randomizer (optional)
    randomizer = None
    if not args.no_domain_rand:
        randomizer = registry.get_domain_randomizer(env_name)

    # Network factory handling (matches notebook pattern)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks

    if "network_factory" in ppo_params:
        # Some configs provide nested network params
        del ppo_training_params["network_factory"]
        net_cfg = dict(ppo_params.network_factory)

        def network_factory(obs_size, act_size, preprocess_observations_fn=None, **kw):
            return ppo_networks.make_ppo_networks(
                obs_size,
                act_size,
                preprocess_observations_fn=preprocess_observations_fn,
                **net_cfg,
            )

    # Progress logging
    csv_path = run_dir / "progress.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env_steps", "eval/episode_reward", "eval/episode_reward_std", "wall_time_sec"],
        )
        writer.writeheader()
        t0 = time.time()

        def progress(num_steps: int, metrics: Dict[str, Any]):
            row = {
                "env_steps": int(num_steps),
                "eval/episode_reward": float(metrics.get("eval/episode_reward", float("nan"))),
                "eval/episode_reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
                "wall_time_sec": float(time.time() - t0),
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{env_name}] steps={row['env_steps']:,} "
                f"reward={row['eval/episode_reward']:.3f} ± {row['eval/episode_reward_std']:.3f} "
                f"t={row['wall_time_sec']:.1f}s"
            )

        make_inference_fn, params, metrics = ppo_train.train(
            environment=env,
            eval_env=registry.load(env_name, config=env_cfg),
            wrap_env_fn=wrapper.wrap_for_brax_training,
            network_factory=network_factory,
            randomization_fn=randomizer,
            progress_fn=progress,
            save_checkpoint_path=str(ckpt_dir),  # <<< ABSOLUTE PATH (fixed)
            seed=int(args.seed),
            **ppo_training_params,
        )

    # Save final metrics snapshot
    (run_dir / "final_metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))

    print("\nDone.")
    print(f"Run dir:      {run_dir}")
    print(f"Checkpoints:  {ckpt_dir}")
    print(f"Progress CSV: {csv_path}")


if __name__ == "__main__":
    main()
