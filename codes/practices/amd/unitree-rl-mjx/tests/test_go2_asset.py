"""The Go2 description compiles and its constants agree with the deployment contract."""

import re

import mujoco
import pytest

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(go2.GO2_XML))


@pytest.fixture(scope="module")
def scene_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(go2.GO2_SCENE_XML))


def test_leg_joints_are_declared_in_the_expected_order(model):
  names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  ]
  # A free joint for the floating base, then three joints per leg.
  assert names[0] == "floating_base_joint"
  assert tuple(names[1:]) == go2.JOINT_ORDER


def test_feet_have_sites_and_collision_geoms(model):
  for foot in go2.FOOT_NAMES:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot) >= 0
    geom = f"{foot}_foot_collision"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0


def test_robot_description_carries_no_actuators_or_terrain(model):
  # Actuators, keyframe and ground plane belong to the environment, not the robot.
  assert model.nu == 0
  assert model.nkey == 0


def test_default_joint_pos_matches_the_deployment_contract():
  # deploy.yaml's `default_joint_pos`, which is also the action offset.
  assert go2.default_joint_pos() == (
    -0.1, 0.9, -1.8,  # FL
    0.1, 0.9, -1.8,  # FR
    -0.1, 0.9, -1.8,  # RL
    0.1, 0.9, -1.8,  # RR
  )  # fmt: skip


def test_actuator_gains_match_the_deployment_contract():
  # deploy.yaml ships stiffness [20, 20, 40] and damping [1, 1, 2] per leg.
  assert (go2.GO2_ACTUATOR_HIP.stiffness, go2.GO2_ACTUATOR_HIP.damping) == (20.0, 1.0)
  assert (go2.GO2_ACTUATOR_THIGH.stiffness, go2.GO2_ACTUATOR_THIGH.damping) == (
    20.0,
    1.0,
  )
  assert (go2.GO2_ACTUATOR_CALF.stiffness, go2.GO2_ACTUATOR_CALF.damping) == (40.0, 2.0)


@pytest.fixture(scope="module")
def training_model() -> mujoco.MjModel:
  return go2.get_go2_training_spec().compile()


def test_training_model_has_one_position_servo_per_joint(training_model):
  names = [
    mujoco.mj_id2name(training_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(training_model.nu)
  ]
  assert tuple(names) == go2.JOINT_ORDER


def test_training_actuator_gains_follow_the_actuator_config(training_model):
  # Position servo: gainprm[0] = kp, biasprm = (0, -kp, -kv).
  for i, joint in enumerate(go2.JOINT_ORDER):
    cfg = next(
      c
      for c in go2.GO2_ACTUATORS
      if any(re.match(e, joint) for e in c.target_names_expr)
    )
    assert training_model.actuator_gainprm[i][0] == cfg.stiffness, joint
    assert training_model.actuator_biasprm[i][1] == -cfg.stiffness, joint
    assert training_model.actuator_biasprm[i][2] == -cfg.damping, joint
    assert tuple(training_model.actuator_forcerange[i]) == (
      -cfg.effort_limit,
      cfg.effort_limit,
    ), joint


def test_training_joint_armature_follows_the_actuator_config(training_model):
  for i, joint in enumerate(go2.JOINT_ORDER):
    expected = 0.02 if "calf" in joint else 0.01
    jid = mujoco.mj_name2id(training_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    assert training_model.dof_armature[training_model.jnt_dofadr[jid]] == expected


def test_training_collision_set_is_all_collision_geoms_plus_terrain(training_model):
  # The reference's velocity task uses FULL_COLLISION: every `*_collision` geom
  # contacts the terrain (feet with contact-rich params, the rest condim 1) so
  # that the non-foot ground-touch termination can fire. Self-collisions stay off.
  colliding = set()
  for i in range(training_model.ngeom):
    if training_model.geom_contype[i] or training_model.geom_conaffinity[i]:
      colliding.add(mujoco.mj_id2name(training_model, mujoco.mjtObj.mjOBJ_GEOM, i))
  collision_geoms = {n for n in colliding if n.endswith("_collision")}
  assert len(collision_geoms) == 23
  assert colliding == collision_geoms | {"terrain"}
  for i in range(training_model.ngeom):
    name = mujoco.mj_id2name(training_model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name in collision_geoms:
      # contype 1 / conaffinity 0 collides with the terrain but not the robot.
      assert training_model.geom_contype[i] == 1
      assert training_model.geom_conaffinity[i] == 0


def test_training_feet_carry_the_reference_contact_params(training_model):
  import numpy as np

  for foot in go2.FOOT_NAMES:
    gid = mujoco.mj_name2id(
      training_model, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_foot_collision"
    )
    assert training_model.geom_condim[gid] == 3
    assert training_model.geom_priority[gid] == 1
    assert training_model.geom_friction[gid][0] == 0.6
    np.testing.assert_allclose(training_model.geom_solimp[gid][:3], (0.9, 0.95, 0.023))


def test_training_nonfoot_collisions_are_frictionless_point_contacts(training_model):
  foot_geoms = {f"{n}_foot_collision" for n in go2.FOOT_NAMES}
  for i in range(training_model.ngeom):
    name = mujoco.mj_id2name(training_model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and name.endswith("_collision") and name not in foot_geoms:
      assert training_model.geom_condim[i] == 1, name


def test_training_keyframe_holds_the_init_pose(training_model):
  import numpy as np

  assert training_model.nkey == 1
  np.testing.assert_allclose(
    training_model.key_qpos[0],
    [*go2.INIT_STATE.pos, 1.0, 0.0, 0.0, 0.0, *go2.default_joint_pos()],
  )
  # ctrl0 = default joint pos: a zero action holds the pose.
  np.testing.assert_allclose(training_model.key_ctrl[0], go2.default_joint_pos())


def test_training_solver_settings_match_the_reference(training_model):
  assert training_model.opt.timestep == 0.005
  assert training_model.opt.iterations == 10
  assert training_model.opt.ls_iterations == 20


def test_scene_is_a_complete_standalone_model(scene_model):
  assert scene_model.nu == 12
  assert scene_model.nkey == 1
  assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_scene_rest_pose_differs_from_the_training_init_state(scene_model):
  # Documented in PROVENANCE.md: the scene's `home` keyframe is not the pose
  # training starts from. Training and deployment both use INIT_STATE.
  assert scene_model.key_qpos[0][2] == pytest.approx(0.27)
  assert go2.INIT_STATE.pos[2] == pytest.approx(0.32)
