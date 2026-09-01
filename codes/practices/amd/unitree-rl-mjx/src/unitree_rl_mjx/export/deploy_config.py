"""Deployment contract for the official Unitree deploy stack.

The deploy runtime loads a policy folder holding `params/deploy.yaml` and
`exported/policy.onnx`. Every deploy.yaml field here is derived from the same
constants the training env uses, so the deployed contract cannot silently
drift from what the policy was trained on.
"""

from __future__ import annotations

import re

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2
from unitree_rl_mjx.tasks.velocity.config.go2.env_cfgs import (
  unitree_go2_flat_env_cfg,
)

SDK_JOINT_ORDER = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
)
"""Go2 motor order in the SDK's low-level command; deploy.yaml's
joint_ids_map holds the training-order index of each SDK slot."""

# Actor observation term -> (deploy.yaml observation name, size, extra params).
# The deploy stack computes each of these on the robot; the names differ from
# the training cfg's keys but the order and sizes must match exactly.
_DEPLOY_OBSERVATIONS = {
  "base_ang_vel": ("base_ang_vel", 3, {}),
  "projected_gravity": ("projected_gravity", 3, {}),
  "command": ("velocity_commands", 3, {"command_name": "base_velocity"}),
  "phase": ("gait_phase", 2, None),  # params filled from the term's period
  "joint_pos": ("joint_pos_rel", 12, {}),
  "joint_vel": ("joint_vel_rel", 12, {}),
  "actions": ("last_action", 12, {}),
}


def _actuator_cfg(joint_name: str) -> go2.ActuatorCfg:
  return next(
    cfg
    for cfg in go2.GO2_ACTUATORS
    if any(re.match(expr, joint_name) for expr in cfg.target_names_expr)
  )


def go2_velocity_deploy_cfg() -> dict:
  """Create the deploy.yaml contents for the Go2 velocity policy."""
  cfg = unitree_go2_flat_env_cfg()
  default_joint_pos = [float(v) for v in go2.default_joint_pos()]
  # Deployment commands stay within the ranges every curriculum stage trained
  # on, i.e. the first stage's.
  stage0 = cfg.curriculum["command_vel"].params["velocity_stages"][0]
  action = cfg.actions["joint_pos"]

  observations = {}
  for term_name, term in cfg.observations["actor"].terms.items():
    deploy_name, size, params = _DEPLOY_OBSERVATIONS[term_name]
    if params is None:
      params = {"period": term.params["period"]}
    observations[deploy_name] = {
      "params": params,
      "clip": None,
      "scale": [1.0] * size,
      "history_length": 1,
    }

  return {
    "joint_ids_map": [go2.JOINT_ORDER.index(name) for name in SDK_JOINT_ORDER],
    "step_dt": cfg.sim.timestep * cfg.decimation,
    "stiffness": [_actuator_cfg(n).stiffness for n in go2.JOINT_ORDER],
    "damping": [_actuator_cfg(n).damping for n in go2.JOINT_ORDER],
    "default_joint_pos": default_joint_pos,
    "commands": {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": list(stage0["lin_vel_x"]),
          "lin_vel_y": list(stage0["lin_vel_y"]),
          "ang_vel_z": list(stage0["ang_vel_z"]),
          "heading": None,
        }
      }
    },
    "actions": {
      "JointPositionAction": {
        "clip": None,
        "joint_names": [".*"],
        "scale": [action.scale] * len(go2.JOINT_ORDER),
        "offset": default_joint_pos,
        "joint_ids": None,
      }
    },
    "observations": observations,
  }
