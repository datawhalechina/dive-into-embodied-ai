from __future__ import annotations

import re

import jax
import jax.numpy as jnp

from unitree_rl_mjx.tasks.velocity.mdp.types import MdpEnv


def _command_activity(env: MdpEnv, command_threshold: float) -> jax.Array:
  """1.0 when the command is active (linear + angular norm above threshold)."""
  linear_norm = jnp.linalg.norm(env.command[:2])
  angular_norm = jnp.abs(env.command[2])
  return (linear_norm + angular_norm > command_threshold).astype(jnp.float32)


def _resolve_joint_values(
  patterns: dict[str, float], joint_names: tuple[str, ...]
) -> jax.Array:
  """Resolve a joint-name-pattern map into one value per joint."""
  values = []
  for name in joint_names:
    matches = [v for expr, v in patterns.items() if re.match(expr, name)]
    if len(matches) != 1:
      raise ValueError(f"{name} matched {len(matches)} patterns, expected 1.")
    values.append(matches[0])
  return jnp.array(values)


def track_linear_velocity(env: MdpEnv, std: float, command_name: str) -> jax.Array:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  del command_name  # Single command term.
  actual = env.root_lin_vel_b
  xy_error = jnp.sum(jnp.square(env.command[:2] - actual[:2]))
  z_error = jnp.square(actual[2])
  lin_vel_error = xy_error + (2 * z_error)
  return jnp.exp(-lin_vel_error / std**2)


def track_angular_velocity(env: MdpEnv, std: float, command_name: str) -> jax.Array:
  """Reward yaw-rate tracking; xy angular velocities are assumed zero."""
  del command_name  # Single command term.
  actual = env.root_ang_vel_b
  z_error = jnp.square(env.command[2] - actual[2])
  xy_error = jnp.sum(jnp.square(actual[:2]))
  ang_vel_error = z_error + (0.05 * xy_error)
  return jnp.exp(-ang_vel_error / std**2)


def body_orientation_l2(env: MdpEnv) -> jax.Array:
  """Reward flat base orientation (robot being upright)."""
  return jnp.sum(jnp.square(env.projected_gravity_b[:2]))


def variable_posture(
  env: MdpEnv,
  std_standing: dict[str, float],
  std_walking: dict[str, float],
  std_running: dict[str, float],
  command_name: str,
  walking_threshold: float = 0.5,
  running_threshold: float = 1.5,
) -> jax.Array:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Per-joint standard deviations pick how much each joint may deviate; the
  speed regime (standing/walking/running, from linear + angular command speed)
  picks which std set applies. Reward is exp(-mean(error^2 / std^2)).
  """
  del command_name  # Single command term.
  stds = jnp.stack(
    [
      _resolve_joint_values(patterns, env.joint_names)
      for patterns in (std_standing, std_walking, std_running)
    ]
  )
  total_speed = jnp.linalg.norm(env.command[:2]) + jnp.abs(env.command[2])
  regime = jnp.where(
    total_speed >= running_threshold,
    2,
    jnp.where(total_speed >= walking_threshold, 1, 0),
  )
  std = stds[regime]
  error_squared = jnp.square(env.joint_pos - env.default_joint_pos)
  return jnp.exp(-jnp.mean(error_squared / jnp.square(std)))


def body_angular_velocity_penalty(env: MdpEnv) -> jax.Array:
  """Penalize excessive body angular velocities."""
  ang_vel_xy = env.root_ang_vel_w[:2]  # Don't penalize z-angular velocity.
  return jnp.sum(jnp.square(ang_vel_xy))


def angular_momentum_penalty(env: MdpEnv, sensor_name: str) -> jax.Array:
  """Penalize whole-body angular momentum."""
  angmom = env.sensors[sensor_name]
  return jnp.sum(jnp.square(angmom))


def is_terminated(env: MdpEnv) -> jax.Array:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.is_terminated.astype(jnp.float32)


def joint_acc_l2(env: MdpEnv) -> jax.Array:
  """Penalize joint accelerations using L2 squared kernel."""
  return jnp.sum(jnp.square(env.joint_acc))


def joint_pos_limits(env: MdpEnv) -> jax.Array:
  """Penalize joint positions if they cross the soft limits."""
  out_of_limits = -jnp.clip(env.joint_pos - env.soft_joint_pos_lower, max=0.0)
  out_of_limits += jnp.clip(env.joint_pos - env.soft_joint_pos_upper, min=0.0)
  return jnp.sum(out_of_limits)


def action_rate_l2(env: MdpEnv) -> jax.Array:
  """Penalize the rate of change of the raw policy actions."""
  return jnp.sum(jnp.square(env.actions - env.last_actions))


def feet_gait(
  env: MdpEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> jax.Array:
  """Reward feet whose contact state matches the commanded gait phase."""
  del command_name, sensor_name  # Single command and foot-contact sensor.
  is_contact = env.current_contact_time > 0
  global_phase = (env.episode_length_buf * env.step_dt) / period
  leg_phase = (global_phase + jnp.array(offset)) % 1.0
  is_stance = leg_phase < threshold
  reward = jnp.mean((is_stance == is_contact).astype(jnp.float32))
  return reward * _command_activity(env, command_threshold)


def feet_clearance(
  env: MdpEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> jax.Array:
  """Penalize deviation from target clearance height, weighted by foot speed."""
  del command_name  # Single command term.
  foot_z = env.foot_pos_w[:, 2]
  vel_norm = jnp.linalg.norm(env.foot_lin_vel_w[:, :2], axis=-1)
  delta = jnp.abs(foot_z - target_height)
  cost = jnp.sum(delta * vel_norm)
  return cost * _command_activity(env, command_threshold)


def feet_slip(
  env: MdpEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
) -> jax.Array:
  """Penalize foot sliding (xy velocity while in contact)."""
  del command_name, sensor_name  # Single command and foot-contact sensor.
  in_contact = env.foot_in_contact.astype(jnp.float32)
  vel_xy_norm_sq = jnp.sum(jnp.square(env.foot_lin_vel_w[:, :2]), axis=-1)
  cost = jnp.sum(vel_xy_norm_sq * in_contact)
  return cost * _command_activity(env, command_threshold)


def soft_landing(
  env: MdpEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> jax.Array:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  del command_name, sensor_name  # Single command and foot-contact sensor.
  force_magnitude = jnp.linalg.norm(env.foot_forces, axis=-1)
  landing_impact = force_magnitude * env.first_contact.astype(jnp.float32)
  cost = jnp.sum(landing_impact)
  return cost * _command_activity(env, command_threshold)


def stand_still(
  env: MdpEnv, command_name: str, command_threshold: float = 0.1
) -> jax.Array:
  """Penalize pose deviation when the command asks for standing still."""
  del command_name  # Single command term.
  diff_angle = env.joint_pos - env.default_joint_pos
  reward = jnp.sum(jnp.square(diff_angle))
  return reward * (1.0 - _command_activity(env, command_threshold))
