"""Frozen `MdpEnv` states for term unit tests.

`make_mdp_env` builds a fully-populated state of zeros (robot at the default
pose, standing still, no contacts) and applies keyword overrides, so each test
states only the fields its term reads.
"""

import jax.numpy as jnp

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2
from unitree_rl_mjx.tasks.velocity.mdp.types import MdpEnv

STEP_DT = 0.02

_DEFAULT_JOINT_POS = jnp.array(go2.default_joint_pos())


def make_mdp_env(**overrides) -> MdpEnv:
  base: dict = dict(  # noqa: C408 - kwargs mirror the dataclass fields
    joint_names=go2.JOINT_ORDER,
    step_dt=STEP_DT,
    episode_length_buf=jnp.zeros((), jnp.int32),
    command=jnp.zeros(3),
    root_lin_vel_b=jnp.zeros(3),
    root_ang_vel_b=jnp.zeros(3),
    root_ang_vel_w=jnp.zeros(3),
    projected_gravity_b=jnp.array([0.0, 0.0, -1.0]),
    joint_pos=_DEFAULT_JOINT_POS,
    joint_vel=jnp.zeros(12),
    joint_acc=jnp.zeros(12),
    default_joint_pos=_DEFAULT_JOINT_POS,
    soft_joint_pos_lower=jnp.full(12, -10.0),
    soft_joint_pos_upper=jnp.full(12, 10.0),
    actions=jnp.zeros(12),
    last_actions=jnp.zeros(12),
    sensors={},
    foot_in_contact=jnp.ones(4, dtype=bool),
    foot_forces=jnp.zeros((4, 3)),
    current_air_time=jnp.zeros(4),
    current_contact_time=jnp.full(4, STEP_DT),
    first_contact=jnp.zeros(4, dtype=bool),
    foot_pos_w=jnp.zeros((4, 3)),
    foot_lin_vel_w=jnp.zeros((4, 3)),
    is_terminated=jnp.zeros((), bool),
  )
  base.update(overrides)
  return MdpEnv(**base)
