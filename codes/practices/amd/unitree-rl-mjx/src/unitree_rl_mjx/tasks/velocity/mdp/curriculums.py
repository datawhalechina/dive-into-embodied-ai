from __future__ import annotations

import jax
import jax.numpy as jnp


def commands_vel(
  common_step_counter: jax.Array,
  velocity_stages: tuple[dict, ...],
  ranges: jax.Array,
) -> jax.Array:
  """Command ranges for the current curriculum stage.

  Takes the configured (lin_vel_x, lin_vel_y, ang_vel_z) ranges as a (3, 2)
  array and returns the ranges with every stage whose step threshold
  `common_step_counter` has passed applied cumulatively, as in the reference;
  a stage may update only some axes.
  """
  for stage in velocity_stages:
    active = common_step_counter > stage["step"]
    for i, axis in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
      if axis in stage and stage[axis] is not None:
        ranges = ranges.at[i].set(
          jnp.where(active, jnp.asarray(stage[axis]), ranges[i])
        )
  return ranges
