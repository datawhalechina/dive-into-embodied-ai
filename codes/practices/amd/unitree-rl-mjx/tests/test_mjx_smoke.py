"""MJX steps the Go2 on CPU without producing non-finite state.

This is the toolchain smoke test: it fails if the jax/mujoco/mjx pins stop working
together, if the asset breaks, or if the physics settings produce NaNs.
"""

import jax
import jax.numpy as jnp
import mujoco
import pytest
from mujoco import mjx

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2

NUM_ENVS = 4
NUM_STEPS = 200
TIMESTEP = go2.SIM_TIMESTEP
START_HEIGHT = go2.INIT_STATE.pos[2]


@pytest.fixture(scope="module")
def batched_rollout():
  """Step the flat-ground model from its init pose, batched, and return the states."""
  mj_model = go2.get_go2_flat_spec().compile()
  assert mj_model.opt.timestep == TIMESTEP
  assert mj_model.key_qpos[0][2] == START_HEIGHT
  model = mjx.put_model(mj_model)

  data = mjx.make_data(mj_model)
  data = data.replace(qpos=jnp.asarray(mj_model.key_qpos[0]))
  batch = jax.tree.map(
    lambda leaf: jnp.broadcast_to(leaf, (NUM_ENVS, *jnp.shape(leaf))), data
  )

  step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
  states = []
  for _ in range(NUM_STEPS):
    batch = step(model, batch)
    states.append(batch)
  return states


def test_jax_runs_on_cpu():
  assert jax.devices()  # Any backend; CI has CPU only.


def test_rollout_state_stays_finite(batched_rollout):
  for i, state in enumerate(batched_rollout):
    assert jnp.all(jnp.isfinite(state.qpos)), f"non-finite qpos at step {i}"
    assert jnp.all(jnp.isfinite(state.qvel)), f"non-finite qvel at step {i}"


def test_rollout_is_batched(batched_rollout):
  assert batched_rollout[-1].qpos.shape[0] == NUM_ENVS


def test_feet_contact_slows_the_fall(batched_rollout):
  """Contact forces are actually being applied, not silently skipped.

  Nothing holds the joints here — the scene's motors are unpowered and only the
  feet collide — so the robot folds up rather than standing. What distinguishes a
  live contact pipeline from a dead one is the fall rate: unopposed gravity would
  drop the base far below where the feet hold it.
  """
  height = batched_rollout[-1].qpos[:, 2]
  free_fall = START_HEIGHT - 0.5 * 9.81 * (NUM_STEPS * TIMESTEP) ** 2
  assert jnp.all(height > free_fall + 0.2), (
    f"base at {height} is close to free fall ({free_fall:.3f}); feet are not colliding"
  )


def test_training_model_stands_under_its_servos():
  """The training spec holds its init pose when driven with the keyframe ctrl.

  Position servos at the default pose should keep the base near its 0.32 m start
  height for a full second of simulation. A wide band (±0.1 m) tolerates the
  initial settle; a folded (< 0.2 m) or launched robot means the actuator gains,
  armature, or contact parameters are wrong.
  """
  mj_model = go2.get_go2_training_spec().compile()
  model = mjx.put_model(mj_model)
  data = mjx.make_data(mj_model).replace(
    qpos=jnp.asarray(mj_model.key_qpos[0]),
    ctrl=jnp.asarray(mj_model.key_ctrl[0]),
  )
  step = jax.jit(mjx.step)
  for _ in range(200):
    data = step(model, data)
  assert jnp.all(jnp.isfinite(data.qpos)) and jnp.all(jnp.isfinite(data.qvel))
  assert 0.22 < data.qpos[2] < 0.42, f"base height {data.qpos[2]}"


def test_mjx_lacks_self_collision_support():
  """Canary: MJX has no cylinder-box collision function.

  The robot's own geometry pairs cylinders (hips, calves) with boxes (base,
  thighs), so any configuration where robot geoms collide with each other cannot
  be put on device. Colliding every geom against the terrain is fine — see
  `test_all_geoms_can_collide_with_terrain`. If MJX gains that collision function
  this test fails, and self-collision becomes available.
  """
  spec = go2.get_go2_spec()
  for geom in spec.geoms:
    if "_collision" in (geom.name or ""):
      geom.contype, geom.conaffinity = 1, 1  # allow robot-vs-robot pairs
  with pytest.raises(NotImplementedError, match="collisions not implemented"):
    mjx.put_model(spec.compile())


def test_all_geoms_can_collide_with_terrain():
  """The reference's full collision set (every geom vs terrain) runs on MJX.

  The shipped task uses feet-only collision for contact cost; this records that
  the reference's richer configuration stays available on MJX.
  """
  spec = go2.get_go2_spec()
  plane = spec.worldbody.add_geom()
  plane.name, plane.type = "terrain", mujoco.mjtGeom.mjGEOM_PLANE
  plane.size = [0, 0, 0.05]
  for geom in spec.geoms:
    if "_collision" in (geom.name or ""):
      geom.contype, geom.conaffinity = 1, 0  # terrain only, no self-collision
    elif geom.name != "terrain":
      geom.contype, geom.conaffinity = 0, 0
  assert mjx.put_model(spec.compile()) is not None
