"""Unitree Go2 velocity environment configurations."""

from __future__ import annotations

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2
from unitree_rl_mjx.tasks.velocity import mdp
from unitree_rl_mjx.tasks.velocity.velocity_env_cfg import (
  TerminationTermCfg,
  VelocityEnvCfg,
  make_velocity_env_cfg,
)


def unitree_go2_flat_env_cfg() -> VelocityEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  nonfoot_sensors = tuple(
    f"{name}_ground_touch" for name in go2.nonfoot_collision_geom_names()
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_names": nonfoot_sensors, "force_threshold": 10.0},
  )

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]

  return cfg
