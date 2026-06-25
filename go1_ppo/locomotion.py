"""Standalone locomotion shim for config compatibility."""

from go1_ppo import registry


def get_default_config(env_name: str):
  return registry.get_default_config(env_name)
