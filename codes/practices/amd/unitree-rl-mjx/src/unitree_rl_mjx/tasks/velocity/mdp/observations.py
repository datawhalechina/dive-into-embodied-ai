from __future__ import annotations

import jax
import jax.numpy as jnp

from unitree_rl_mjx.tasks.velocity.mdp.types import MdpEnv


def builtin_sensor(env: MdpEnv, sensor_name: str) -> jax.Array:
  return env.sensors[sensor_name]


def projected_gravity(env: MdpEnv) -> jax.Array:
  return env.projected_gravity_b


def generated_commands(env: MdpEnv, command_name: str) -> jax.Array:
  del command_name  # Single command term.
  return env.command


def phase(env: MdpEnv, period: float, command_name: str) -> jax.Array:
  """Gait phase clock as (sin, cos), zeroed for standing commands."""
  del command_name  # Single command term.
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  p = jnp.array(
    [jnp.sin(global_phase * jnp.pi * 2.0), jnp.cos(global_phase * jnp.pi * 2.0)]
  )
  stand = jnp.linalg.norm(env.command) < 0.1
  return jnp.where(stand, jnp.zeros_like(p), p)


def joint_pos_rel(env: MdpEnv) -> jax.Array:
  return env.joint_pos - env.default_joint_pos


def joint_vel_rel(env: MdpEnv) -> jax.Array:
  # The default joint velocity is zero.
  return env.joint_vel


def last_action(env: MdpEnv) -> jax.Array:
  return env.actions


def foot_height(env: MdpEnv) -> jax.Array:
  return env.foot_pos_w[:, 2]


def foot_air_time(env: MdpEnv) -> jax.Array:
  return env.current_air_time


def foot_contact(env: MdpEnv) -> jax.Array:
  return env.foot_in_contact.astype(jnp.float32)


def foot_contact_forces(env: MdpEnv) -> jax.Array:
  forces = env.foot_forces.ravel()
  return jnp.sign(forces) * jnp.log1p(jnp.abs(forces))
