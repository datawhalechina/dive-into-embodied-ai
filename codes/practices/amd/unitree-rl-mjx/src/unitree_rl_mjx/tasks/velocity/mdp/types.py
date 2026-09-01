"""Per-step state view consumed by mdp term functions.

The environment assembles one `MdpEnv` per control step; observation, reward,
and termination terms read from it. Terms take it as a parameter named `env`
so their bodies read side by side with the reference implementation, and tests
can build one directly from a frozen state without stepping physics.
"""

from __future__ import annotations

import jax
from flax import struct


@struct.dataclass
class MdpEnv:
  """Unbatched view of one environment at one control step."""

  joint_names: tuple[str, ...] = struct.field(pytree_node=False)
  step_dt: float
  episode_length_buf: jax.Array
  command: jax.Array
  root_lin_vel_b: jax.Array
  root_ang_vel_b: jax.Array
  root_ang_vel_w: jax.Array
  projected_gravity_b: jax.Array
  joint_pos: jax.Array
  joint_vel: jax.Array
  joint_acc: jax.Array
  default_joint_pos: jax.Array
  soft_joint_pos_lower: jax.Array
  soft_joint_pos_upper: jax.Array
  actions: jax.Array
  last_actions: jax.Array
  sensors: dict[str, jax.Array]
  foot_in_contact: jax.Array
  foot_forces: jax.Array
  current_air_time: jax.Array
  current_contact_time: jax.Array
  first_contact: jax.Array
  foot_pos_w: jax.Array
  foot_lin_vel_w: jax.Array
  is_terminated: jax.Array
