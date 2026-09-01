from __future__ import annotations

import jax
import jax.numpy as jnp

from unitree_rl_mjx.tasks.velocity.mdp.types import MdpEnv


def time_out(env: MdpEnv) -> jax.Array:
  """Episode truncation, applied by the training-time episode wrapper.

  The config lists it alongside the other terminations; inside the env it
  never fires, so timeouts are truncations (bootstrapped), not failures.
  """
  del env  # Unused.
  return jnp.zeros((), dtype=bool)


def bad_orientation(env: MdpEnv, limit_angle: float) -> jax.Array:
  """Terminate when the base tilts past `limit_angle` from upright."""
  # projected_gravity_b[2] is -1 upright, 0 sideways; acos of its negation is
  # the tilt angle.
  tilt = jnp.arccos(-jnp.clip(env.projected_gravity_b[2], -1.0, 1.0))
  return tilt > limit_angle


def illegal_contact(
  env: MdpEnv, sensor_names: tuple[str, ...], force_threshold: float = 10.0
) -> jax.Array:
  """Terminate when any listed contact sensor reports force above threshold.

  MJX contact sensors report the last substep only, so the check is
  instantaneous at the control rate rather than over a substep history.
  """
  hit = jnp.array(
    [jnp.linalg.norm(env.sensors[name][1:4]) > force_threshold for name in sensor_names]
  )
  return jnp.any(hit)
