from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx


def domain_randomize(
  model: mjx.Model,
  rng: jax.Array,
  foot_geom_ids: tuple[int, ...],
  base_body_id: int,
  friction_range: tuple[float, float],
  com_offset_ranges,
):
  """Startup model-field randomization for a batch of environments.

  Per environment: one shared tangential friction for all foot geoms
  (absolute, uniform in `friction_range`) and an additive base CoM offset
  (uniform per axis in `com_offset_ranges`, a (3, 2) array of bounds). Returns the batched model and the
  matching vmap in_axes, the interface brax's randomization hook expects.
  """
  foot_ids = jnp.array(foot_geom_ids)
  com_ranges = jnp.asarray(com_offset_ranges)

  @jax.vmap
  def rand(key: jax.Array):
    k_friction, k_com = jax.random.split(key)
    friction = jax.random.uniform(
      k_friction, minval=friction_range[0], maxval=friction_range[1]
    )
    geom_friction = model.geom_friction.at[foot_ids, 0].set(friction)
    com_offset = jax.random.uniform(
      k_com, (3,), minval=com_ranges[:, 0], maxval=com_ranges[:, 1]
    )
    body_ipos = model.body_ipos.at[base_body_id].add(com_offset)
    return geom_friction, body_ipos

  geom_friction, body_ipos = rand(rng)
  in_axes = jax.tree.map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_friction": 0, "body_ipos": 0})
  model = model.tree_replace({"geom_friction": geom_friction, "body_ipos": body_ipos})
  return model, in_axes
