"""Per-term unit tests for the velocity task rewards on frozen states.

Every test evaluates the real term function against a hand-computed value.
Tolerances are 1e-6 relative unless noted: the terms are closed-form
expressions of the frozen state, so only float32 rounding separates the
implementation from the hand calculation.
"""

import jax.numpy as jnp
import pytest
from mdp_test_utils import make_mdp_env

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2
from unitree_rl_mjx.tasks.velocity import mdp

TOL = 1e-6
WALK_CMD = jnp.array([1.0, 0.0, 0.0])

##
# Tracking terms
##


def test_track_linear_velocity():
  env = make_mdp_env(
    command=jnp.array([1.0, 0.0, 0.5]),
    root_lin_vel_b=jnp.array([0.5, 0.0, 0.2]),
  )
  # xy error 0.25, z error 0.04 doubled: exp(-0.33 / 0.25).
  expected = jnp.exp(-1.32)
  value = mdp.track_linear_velocity(
    env, std=float(jnp.sqrt(0.25)), command_name="twist"
  )
  assert value == pytest.approx(expected, rel=TOL)


def test_track_angular_velocity():
  env = make_mdp_env(
    command=jnp.array([0.0, 0.0, 0.5]),
    root_ang_vel_b=jnp.array([0.1, 0.2, 0.3]),
  )
  # z error 0.04, xy 0.05 damped by 0.05: exp(-0.0425 / 0.5).
  expected = jnp.exp(-0.085)
  value = mdp.track_angular_velocity(
    env, std=float(jnp.sqrt(0.5)), command_name="twist"
  )
  assert value == pytest.approx(expected, rel=TOL)


##
# Posture & stability
##


def test_body_orientation_l2():
  env = make_mdp_env(projected_gravity_b=jnp.array([0.3, -0.4, -0.866]))
  value = mdp.body_orientation_l2(env)
  assert value == pytest.approx(0.25, rel=TOL)  # 0.09 + 0.16


def test_variable_posture_uses_the_walking_std():
  # Speed 0.8 sits between the Go2 thresholds (0.1, 1.5): walking regime.
  env = make_mdp_env(
    command=jnp.array([0.5, 0.0, 0.3]),
    joint_pos=jnp.array(go2.default_joint_pos()) + 0.1,
  )
  stds = {".*": 0.5}
  # Per-joint error 0.01 / 0.25 = 0.04: exp(-0.04).
  value = mdp.variable_posture(
    env,
    std_standing={".*": 0.05},
    std_walking=stds,
    std_running=stds,
    command_name="twist",
    walking_threshold=0.1,
    running_threshold=1.5,
  )
  assert value == pytest.approx(jnp.exp(-0.04), rel=TOL)


def test_variable_posture_uses_the_standing_std_below_threshold():
  env = make_mdp_env(
    command=jnp.zeros(3),
    joint_pos=jnp.array(go2.default_joint_pos()) + 0.1,
  )
  # Per-joint error 0.01 / 0.0025 = 4: exp(-4).
  value = mdp.variable_posture(
    env,
    std_standing={".*": 0.05},
    std_walking={".*": 0.5},
    std_running={".*": 0.5},
    command_name="twist",
    walking_threshold=0.1,
    running_threshold=1.5,
  )
  # rel 1e-5: the float32 divide/mean/exp chain rounds ~1e-6 past TOL here.
  assert value == pytest.approx(jnp.exp(-4.0), rel=1e-5)


def test_body_angular_velocity_penalty_ignores_yaw():
  env = make_mdp_env(root_ang_vel_w=jnp.array([0.3, 0.4, 1.0]))
  value = mdp.body_angular_velocity_penalty(env)
  assert value == pytest.approx(0.25, rel=TOL)


def test_angular_momentum_penalty():
  env = make_mdp_env(sensors={"root_angmom": jnp.array([1.0, 2.0, 2.0])})
  value = mdp.angular_momentum_penalty(env, sensor_name="root_angmom")
  assert value == pytest.approx(9.0, rel=TOL)


def test_stand_still_penalizes_pose_error_only_when_standing():
  moved = jnp.array(go2.default_joint_pos()) + 0.1
  standing = make_mdp_env(command=jnp.array([0.05, 0.0, 0.0]), joint_pos=moved)
  walking = make_mdp_env(command=WALK_CMD, joint_pos=moved)
  assert mdp.stand_still(standing, command_name="twist") == pytest.approx(
    0.12, rel=1e-5
  )  # 12 joints x 0.01
  assert mdp.stand_still(walking, command_name="twist") == 0.0


##
# Regularizers
##


def test_is_terminated():
  assert mdp.is_terminated(make_mdp_env(is_terminated=jnp.ones((), bool))) == 1.0
  assert mdp.is_terminated(make_mdp_env()) == 0.0


def test_joint_acc_l2():
  env = make_mdp_env(joint_acc=jnp.full(12, 0.5))
  assert mdp.joint_acc_l2(env) == pytest.approx(3.0, rel=TOL)


def test_joint_pos_limits_sums_soft_limit_violations():
  joint_pos = jnp.zeros(12).at[0].set(1.3).at[1].set(-1.2)
  env = make_mdp_env(
    joint_pos=joint_pos,
    soft_joint_pos_lower=jnp.full(12, -1.0),
    soft_joint_pos_upper=jnp.full(12, 1.0),
  )
  # 0.3 over the upper limit + 0.2 under the lower limit.
  assert mdp.joint_pos_limits(env) == pytest.approx(0.5, rel=1e-5)


def test_action_rate_l2():
  env = make_mdp_env(actions=jnp.full(12, 0.1), last_actions=jnp.zeros(12))
  assert mdp.action_rate_l2(env) == pytest.approx(0.12, rel=1e-5)


##
# Gait & feet terms
##


def _gait_env(contact_time, command=WALK_CMD, step=9):
  # Step 9 at 50 Hz is t = 0.18 s: phase 0.3 of the 0.6 s period, so with
  # offsets (0, .5, .5, 0) the diagonal pairs split into stance (0.3 < 0.56)
  # and swing (0.8 > 0.56).
  return make_mdp_env(
    episode_length_buf=jnp.array(step, jnp.int32),
    command=command,
    current_contact_time=jnp.array(contact_time),
  )


GAIT_PARAMS = dict(  # noqa: C408 - mirrors the cfg params
  period=0.6,
  offset=[0.0, 0.5, 0.5, 0.0],
  threshold=0.56,
  command_threshold=0.1,
  command_name="twist",
  sensor_name="feet_ground_contact",
)


def test_feet_gait_rewards_the_trot_pattern():
  dt = 0.02
  trotting = _gait_env([dt, 0.0, 0.0, dt])  # Contacts match stance legs.
  assert mdp.feet_gait(trotting, **GAIT_PARAMS) == pytest.approx(1.0)


def test_feet_gait_scores_half_for_standing_on_all_fours():
  dt = 0.02
  standing = _gait_env([dt] * 4)  # Two legs should be swinging.
  assert mdp.feet_gait(standing, **GAIT_PARAMS) == pytest.approx(0.5)


def test_feet_gait_gated_off_for_small_commands():
  dt = 0.02
  env = _gait_env([dt, 0.0, 0.0, dt], command=jnp.array([0.05, 0.0, 0.0]))
  assert mdp.feet_gait(env, **GAIT_PARAMS) == 0.0


def test_feet_clearance_weights_height_error_by_foot_speed():
  env = make_mdp_env(
    command=WALK_CMD,
    foot_pos_w=jnp.array(
      [[0, 0, 0.05], [0, 0, 0.10], [0, 0, 0.15], [0, 0, 0.02]], jnp.float32
    ),
    foot_lin_vel_w=jnp.array([[1, 0, 9], [2, 0, 9], [0, 2, 9], [0, 0, 5]], jnp.float32),
  )
  # |z - 0.1| * |v_xy|: 0.05*1 + 0*2 + 0.05*2 + 0.08*0 = 0.15.
  value = mdp.feet_clearance(
    env, target_height=0.10, command_name="twist", command_threshold=0.1
  )
  assert value == pytest.approx(0.15, rel=1e-5)


def test_feet_slip_penalizes_xy_speed_of_contact_feet():
  env = make_mdp_env(
    command=WALK_CMD,
    foot_in_contact=jnp.array([True, True, False, False]),
    foot_lin_vel_w=jnp.array(
      [[0.3, 0.4, 9], [0.1, 0, 3], [5, 5, 5], [1, 1, 1]], jnp.float32
    ),
  )
  # v_xy^2 of the two contact feet: 0.25 + 0.01.
  value = mdp.feet_slip(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.1,
  )
  assert value == pytest.approx(0.26, rel=1e-5)


def test_soft_landing_penalizes_first_contact_impacts():
  env = make_mdp_env(
    command=WALK_CMD,
    first_contact=jnp.array([True, True, False, False]),
    foot_forces=jnp.array([[0, 0, 10], [3, 4, 0], [6, 8, 0], [1, 1, 1]], jnp.float32),
  )
  # Force magnitudes 10 and 5 on the two landing feet.
  value = mdp.soft_landing(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.1,
  )
  assert value == pytest.approx(15.0, rel=1e-5)
