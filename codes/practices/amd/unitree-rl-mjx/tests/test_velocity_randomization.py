"""Domain randomization and curriculum: seeds vary, values stay in range."""

import jax
import jax.numpy as jnp
import pytest

from unitree_rl_mjx.envs import Go2VelocityFlat
from unitree_rl_mjx.tasks.velocity import mdp


@pytest.fixture(scope="module")
def env() -> Go2VelocityFlat:
  return Go2VelocityFlat()


@pytest.fixture(scope="module")
def randomized(env):
  rng = jax.random.split(jax.random.PRNGKey(0), 8)
  model, in_axes = env.domain_randomize_fn()(env.mjx_model, rng)
  return model, in_axes


def test_foot_friction_is_shared_resampled_and_in_range(env, randomized):
  model, _ = randomized
  foot_ids = [
    env.mj_model.geom(f"{n}_foot_collision").id for n in ("FR", "FL", "RR", "RL")
  ]
  friction = model.geom_friction[:, foot_ids, 0]  # [B, 4]
  # Shared across the four feet within an env, different across envs.
  assert jnp.allclose(friction, friction[:, :1])
  assert len(jnp.unique(friction[:, 0])) == friction.shape[0]
  assert jnp.all((friction >= 0.3) & (friction <= 1.6))


def test_base_com_offset_is_in_range_and_varies(env, randomized):
  model, _ = randomized
  base_id = env.mj_model.body("base_link").id
  offsets = model.body_ipos[:, base_id] - env.mjx_model.body_ipos[base_id]
  assert jnp.all(jnp.abs(offsets) <= 0.05 + 1e-6)
  assert not jnp.allclose(offsets[0], offsets[1])


def test_randomization_leaves_other_fields_unbatched(randomized):
  model, in_axes = randomized
  assert model.geom_friction.ndim == 3  # batched
  assert model.body_ipos.ndim == 3  # batched
  assert model.body_mass.ndim == 1  # untouched
  assert in_axes.geom_friction == 0
  assert in_axes.body_mass is None


def test_encoder_bias_is_sampled_per_reset_and_biases_joint_obs(env):
  s0 = jax.jit(env.reset)(jax.random.PRNGKey(1))
  s1 = jax.jit(env.reset)(jax.random.PRNGKey(2))
  bias = s0.info["encoder_bias"]
  assert jnp.all(jnp.abs(bias) <= 0.015)
  assert not jnp.allclose(bias, s1.info["encoder_bias"])
  # The critic's joint_pos slice (indices 11:23) carries the bias.
  joint_pos_obs = s0.obs["privileged_state"][11:23]
  true_rel = s0.data.qpos[7:] - env._default_joint_pos
  assert jnp.allclose(joint_pos_obs - true_rel, bias, atol=1e-6)


##
# Curriculum stages (mdp.commands_vel)
##

STAGES = (
  {
    "step": 0,
    "lin_vel_x": (-0.5, 1.0),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-1.0, 1.0),
  },
  {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
)
CFG_RANGES = jnp.array([(-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)])


def test_curriculum_narrows_after_the_first_step():
  # As in the reference, stage 0 applies once the counter passes its step
  # threshold; the very first resample (counter 0) still uses the cfg ranges.
  at_zero = mdp.commands_vel(jnp.array(0), STAGES, CFG_RANGES)
  assert jnp.allclose(at_zero, CFG_RANGES)
  at_one = mdp.commands_vel(jnp.array(1), STAGES, CFG_RANGES)
  assert jnp.allclose(at_one, jnp.array([(-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0)]))


def test_curriculum_widens_past_the_stage_threshold():
  before = mdp.commands_vel(jnp.array(5000 * 24), STAGES, CFG_RANGES)
  assert jnp.allclose(before[0], jnp.array([-0.5, 1.0]))
  after = mdp.commands_vel(jnp.array(5000 * 24 + 1), STAGES, CFG_RANGES)
  assert jnp.allclose(after, jnp.array([(-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)]))
