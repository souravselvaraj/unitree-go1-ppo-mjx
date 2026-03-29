"""Registry for standalone Go1 locomotion environments."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from go1_ppo import mjx_env
from go1_ppo.envs import joystick as go1_joystick
from go1_ppo.envs import randomize as go1_randomize


_envs = {
    "Go1JoystickFlatTerrain": functools.partial(
        go1_joystick.Joystick, task="flat_terrain"
    ),
    "Go1JoystickRoughTerrain": functools.partial(
        go1_joystick.Joystick, task="rough_terrain"
    ),
}

_cfgs = {
    "Go1JoystickFlatTerrain": go1_joystick.default_config,
    "Go1JoystickRoughTerrain": go1_joystick.default_config,
}

_randomizer = {
    "Go1JoystickFlatTerrain": go1_randomize.domain_randomize,
    "Go1JoystickRoughTerrain": go1_randomize.domain_randomize,
}


def __getattr__(name):
  if name == "ALL_ENVS":
    return tuple(_envs.keys())
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def register_environment(
    env_name: str,
    env_class: Type[mjx_env.MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
) -> None:
  """Register a new environment."""
  _envs[env_name] = env_class
  _cfgs[env_name] = cfg_class


def get_default_config(env_name: str) -> config_dict.ConfigDict:
  """Get the default configuration for an environment."""
  if env_name not in _cfgs:
    raise ValueError(
        f"Env '{env_name}' not found in default configs. Available configs:"
        f" {list(_cfgs.keys())}"
    )
  return _cfgs[env_name]()


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
  """Get an environment instance with the given configuration."""
  if env_name not in _envs:
    raise ValueError(
        f"Env '{env_name}' not found. Available envs: {tuple(_envs.keys())}"
    )
  config = config or get_default_config(env_name)
  return _envs[env_name](config=config, config_overrides=config_overrides)


def get_domain_randomizer(
    env_name: str,
) -> Optional[Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]]:
  """Get the default domain randomizer for an environment."""
  if env_name not in _randomizer:
    print(
        f"Env '{env_name}' does not have a domain randomizer in the"
        " standalone registry."
    )
    return None
  return _randomizer[env_name]
