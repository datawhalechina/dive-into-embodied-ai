"""Go2VelocityFlat core: reset, stepping, observation groups, gait phase."""

import jax
import jax.numpy as jnp
import pytest
from mdp_test_utils import make_mdp_env

from unitree_rl_mjx.envs import Go2VelocityFlat
from unitree_rl_mjx.tasks.velocity import mdp

# Actor terms in deploy.yaml order with their widths.
ACTOR_LAYOUT = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("command", 3),
  ("phase", 2),
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("actions", 12),
)
ACTOR_SIZE = sum(w for _, w in ACTOR_LAYOUT)  # 47
PRIVILEGED_SIZE = ACTOR_SIZE + 3 + 4 + 4 + 4 + 12  # 74


def _slice(name: str) -> slice:
  start = 0
  for term, width in ACTOR_LAYOUT:
    if term == name:
      return slice(start, start + width)
    start += width
  raise KeyError(name)


@pytest.fixture(scope="module")
def env() -> Go2VelocityFlat:
  return Go2VelocityFlat()


@pytest.fixture(scope="module")
def reset_state(env):
  return jax.jit(env.reset)(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def stepped_state(env, reset_state):
  state = reset_state
  step = jax.jit(env.step)
  for _ in range(10):
    state = step(state, jnp.zeros(env.action_size))
  return state


def test_actor_terms_follow_the_deploy_contract(env):
  # deploy.yaml lists the policy inputs in this exact order.
  actor = env._cfg.observations["actor"]
  assert tuple(actor.terms.keys()) == tuple(name for name, _ in ACTOR_LAYOUT)


def test_observation_group_sizes(reset_state):
  assert reset_state.obs["state"].shape == (ACTOR_SIZE,)
  assert reset_state.obs["privileged_state"].shape == (PRIVILEGED_SIZE,)


def test_control_rate_and_episode_length(env):
  assert env.dt == pytest.approx(0.02)  # deploy.yaml step_dt
  assert env.sim_dt == pytest.approx(0.005)
  assert env.n_substeps == 4
  assert env._config.episode_length == 1000  # 20 s at 50 Hz


def test_critic_exposes_the_true_state(env, stepped_state):
  # The critic group repeats the actor terms without corruption, so its slices
  # must equal the underlying state exactly.
  critic = stepped_state.obs["privileged_state"]
  data = stepped_state.data
  joint_pos = critic[_slice("joint_pos")]
  # joint_pos reads carry the per-env encoder bias, as in the reference.
  expected = data.qpos[7:] + stepped_state.info["encoder_bias"] - env._default_joint_pos
  assert jnp.allclose(joint_pos, expected, atol=1e-6)
  command = critic[_slice("command")]
  assert jnp.allclose(command, stepped_state.info["command"])
  actions = critic[_slice("actions")]
  assert jnp.allclose(actions, stepped_state.info["actions"])


def test_noise_corrupts_only_the_actor_group(stepped_state):
  actor = stepped_state.obs["state"]
  critic = stepped_state.obs["privileged_state"]
  # Noise-free terms agree between the groups...
  for term in ("command", "phase", "actions"):
    assert jnp.allclose(actor[_slice(term)], critic[_slice(term)])
  # ...noisy terms do not (uniform noise is nonzero almost surely).
  for term in ("base_ang_vel", "joint_pos", "joint_vel"):
    assert not jnp.allclose(actor[_slice(term)], critic[_slice(term)])


REFERENCE_REWARD_KEYS = {
  "track_linear_velocity",
  "track_angular_velocity",
  "body_orientation_l2",
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "is_terminated",
  "joint_acc_l2",
  "joint_pos_limits",
  "action_rate_l2",
  "foot_gait",
  "foot_clearance",
  "foot_slip",
  "soft_landing",
  "stand_still",
}


def test_reward_terms_mirror_the_reference_cfg(env):
  assert set(env._cfg.rewards.keys()) == REFERENCE_REWARD_KEYS


def test_reward_is_finite_and_metrics_track_every_term(stepped_state):
  assert jnp.isfinite(stepped_state.reward)
  for key in REFERENCE_REWARD_KEYS:
    assert jnp.isfinite(stepped_state.metrics[f"reward/{key}"]), key


def test_step_advances_and_stays_finite(reset_state, stepped_state):
  assert stepped_state.info["episode_length_buf"] == 10
  assert jnp.all(jnp.isfinite(stepped_state.obs["state"]))
  assert jnp.all(jnp.isfinite(stepped_state.obs["privileged_state"]))
  assert jnp.all(jnp.isfinite(stepped_state.data.qpos))


def test_standing_env_has_zero_command_and_phase(env, reset_state):
  state = reset_state.replace(
    info={**reset_state.info, "is_standing": jnp.ones((), bool)}
  )
  state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
  assert jnp.allclose(state.obs["privileged_state"][_slice("command")], 0.0)
  assert jnp.allclose(state.obs["privileged_state"][_slice("phase")], 0.0)


def test_reset_randomizes_pose_within_the_configured_ranges(env):
  states = [jax.jit(env.reset)(jax.random.PRNGKey(seed)) for seed in range(4)]
  xy = jnp.stack([s.data.qpos[0:2] for s in states])
  assert not jnp.allclose(xy[0], xy[1])
  assert jnp.all(jnp.abs(xy) <= 0.5 + 1e-6)
  heights = jnp.stack([s.data.qpos[2] for s in states])
  assert jnp.allclose(heights, 0.32)


##
# Gait phase (mdp.phase)
##


def test_phase_is_sin_cos_of_the_gait_clock():
  # Half a 0.6 s period is 15 control steps at 50 Hz: sin ~ 0, cos = -1.
  env = make_mdp_env(
    episode_length_buf=jnp.array(15, jnp.int32),
    command=jnp.array([1.0, 0.0, 0.0]),
  )
  value = mdp.phase(env, period=0.6, command_name="twist")
  assert value == pytest.approx((0.0, -1.0), abs=1e-6)


def test_phase_starts_at_cos_one():
  env = make_mdp_env(command=jnp.array([1.0, 0.0, 0.0]))
  value = mdp.phase(env, period=0.6, command_name="twist")
  assert value == pytest.approx((0.0, 1.0), abs=1e-6)


def test_phase_zeroed_for_standing_commands():
  env = make_mdp_env(
    episode_length_buf=jnp.array(7, jnp.int32),
    command=jnp.array([0.05, 0.0, 0.0]),  # Below the 0.1 standing threshold.
  )
  value = mdp.phase(env, period=0.6, command_name="twist")
  assert value == pytest.approx((0.0, 0.0))


##
# Terminations
##


def test_bad_orientation_fires_past_the_limit_angle():
  upright = make_mdp_env()
  tipped = make_mdp_env(projected_gravity_b=jnp.array([0.94, 0.0, -0.34]))
  limit = jnp.radians(70.0)
  assert not mdp.bad_orientation(upright, limit_angle=limit)
  assert mdp.bad_orientation(tipped, limit_angle=limit)  # ~70.1 deg tilt.


def test_illegal_contact_thresholds_on_force():
  env = make_mdp_env(
    sensors={
      "base1_collision_ground_touch": jnp.array([1.0, 12.0, 0.0, 0.0]),
      "base2_collision_ground_touch": jnp.zeros(4),
    }
  )
  names = ("base1_collision_ground_touch", "base2_collision_ground_touch")
  assert mdp.illegal_contact(env, sensor_names=names, force_threshold=10.0)
  assert not mdp.illegal_contact(env, sensor_names=names, force_threshold=15.0)
