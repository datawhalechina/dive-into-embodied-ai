"""Unitree Go2 constants."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from unitree_rl_mjx import PACKAGE_PATH

##
# MJCF and assets.
##

GO2_XML: Path = PACKAGE_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
"""Robot description. Carries no actuators, keyframe or terrain; those are added
when an environment is assembled."""

GO2_SCENE_XML: Path = (
  PACKAGE_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "scene_go2.xml"
)
"""Standalone scene: robot with torque motors, a ground plane and a home keyframe.
Used for visualization and sim2sim, not for training."""

assert GO2_XML.exists()
assert GO2_SCENE_XML.exists()


def get_go2_spec() -> mujoco.MjSpec:
  """Load the Go2 robot description."""
  return mujoco.MjSpec.from_file(str(GO2_XML))


def get_go2_scene_spec() -> mujoco.MjSpec:
  """Load the standalone Go2 scene."""
  return mujoco.MjSpec.from_file(str(GO2_SCENE_XML))


##
# Actuator config.
##


@dataclass(frozen=True)
class ActuatorCfg:
  """Position-servo parameters shared by a group of joints."""

  target_names_expr: tuple[str, ...]
  stiffness: float
  damping: float
  effort_limit: float
  armature: float


GO2_ACTUATOR_HIP = ActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = ActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = ActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45.0,
  armature=0.02,
)

GO2_ACTUATORS = (GO2_ACTUATOR_HIP, GO2_ACTUATOR_THIGH, GO2_ACTUATOR_CALF)

SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
"""Fraction of each joint's range treated as its soft limit."""

##
# Naming.
##

FOOT_NAMES = ("FR", "FL", "RR", "RL")
"""Foot sites and collision-geom prefixes."""

JOINT_ORDER = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
"""Joint order as declared in the MJCF. Policies observe and act in this order;
the deployment SDK uses its own order and is bridged by a joint index map."""

##
# Keyframes.
##


@dataclass(frozen=True)
class InitialStateCfg:
  """Robot state at episode start."""

  pos: tuple[float, float, float]
  joint_pos: dict[str, float]
  joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})


INIT_STATE = InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)


def default_joint_pos() -> tuple[float, ...]:
  """Resolve `INIT_STATE.joint_pos` patterns into one value per joint.

  Returned in `JOINT_ORDER`. This vector is also the action offset: a policy
  output of zero holds this pose.
  """
  values = []
  for name in JOINT_ORDER:
    matches = [v for expr, v in INIT_STATE.joint_pos.items() if re.match(expr, name)]
    if len(matches) != 1:
      raise ValueError(f"{name} matched {len(matches)} patterns, expected exactly 1.")
    values.append(matches[0])
  return tuple(values)


##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# Keys mirror the reference's collision config fields; a value may be a scalar or a
# mapping from geom-name pattern to value.

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = {
  "geom_names_expr": (_foot_regex,),
  "contype": 0,
  "conaffinity": 1,
  "condim": 3,
  "priority": 1,
  "friction": (0.6,),
  "solimp": (0.9, 0.95, 0.023),
}

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = {
  "geom_names_expr": (".*_collision",),
  "condim": {_foot_regex: 3, ".*_collision": 1},
  "priority": {_foot_regex: 1},
  "friction": {_foot_regex: (0.6,)},
  "solimp": {_foot_regex: (0.9, 0.95, 0.023)},
  "contype": 1,
  "conaffinity": 0,
}


def nonfoot_collision_geom_names() -> tuple[str, ...]:
  """Collision geoms other than the feet, in declaration order."""
  feet = {f"{name}_foot_collision" for name in FOOT_NAMES}
  return tuple(
    geom.name
    for geom in get_go2_spec().geoms
    if geom.name.endswith("_collision") and geom.name not in feet
  )


SIM_TIMESTEP = 0.005
"""Physics timestep. With decimation 4 this gives control at 50 Hz."""

SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20


def _resolve(value, geom_name: str):
  """Resolve a scalar-or-pattern-map collision value for one geom.

  Pattern maps resolve first-match-wins, so specific patterns (the feet) come
  before catch-alls in the config dicts.
  """
  if isinstance(value, dict):
    for expr, v in value.items():
      if re.match(expr, geom_name):
        return v
    return None
  return value


def _apply_collision(spec: mujoco.MjSpec, collision: dict) -> None:
  """Apply a collision config to every geom it matches; leave the rest alone."""
  exprs = collision["geom_names_expr"]
  for geom in spec.geoms:
    if not geom.name or not any(re.match(e, geom.name) for e in exprs):
      continue
    for attr in ("contype", "conaffinity", "condim", "priority"):
      value = _resolve(collision.get(attr), geom.name)
      if value is not None:
        setattr(geom, attr, value)
    friction = _resolve(collision.get("friction"), geom.name)
    if friction is not None:
      geom.friction[: len(friction)] = friction
    solimp = _resolve(collision.get("solimp"), geom.name)
    if solimp is not None:
      geom.solimp[: len(solimp)] = solimp


def _add_position_actuators(spec: mujoco.MjSpec) -> None:
  """One position servo per joint, in `JOINT_ORDER`, with armature on the joint."""
  joints = {joint.name: joint for joint in spec.joints}
  for name in JOINT_ORDER:
    cfg = next(
      c for c in GO2_ACTUATORS if any(re.match(e, name) for e in c.target_names_expr)
    )
    joints[name].armature = cfg.armature
    act = spec.add_actuator()
    act.name = name
    act.target = name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = cfg.stiffness
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm[:3] = (0.0, -cfg.stiffness, -cfg.damping)
    act.forcerange = (-cfg.effort_limit, cfg.effort_limit)


def get_go2_flat_spec(timestep: float = SIM_TIMESTEP) -> mujoco.MjSpec:
  """Build the Go2 on flat ground: the robot as training will assemble it.

  Takes the robot description and adds what an environment supplies — a ground
  plane, the feet-only contact configuration, the physics timestep and the initial
  pose. Deliberately built from `go2.xml` rather than the standalone scene, whose
  contact parameters and integrator settings differ from the ones we train with.

  No actuators: those arrive with the task. Stepping this model with no control
  makes the robot fold up under gravity, which is correct.
  """
  spec = get_go2_spec()
  spec.option.timestep = timestep

  terrain = spec.worldbody.add_geom()
  terrain.name = "terrain"
  terrain.type = mujoco.mjtGeom.mjGEOM_PLANE
  terrain.size = [0, 0, 0.05]

  foot_geoms = {f"{name}_foot_collision" for name in FOOT_NAMES}
  for geom in spec.geoms:
    if geom.name in foot_geoms:
      geom.contype = FEET_ONLY_COLLISION["contype"]
      geom.conaffinity = FEET_ONLY_COLLISION["conaffinity"]
      geom.condim = FEET_ONLY_COLLISION["condim"]
      geom.priority = FEET_ONLY_COLLISION["priority"]
      geom.friction[0] = FEET_ONLY_COLLISION["friction"][0]
      geom.solimp[:3] = FEET_ONLY_COLLISION["solimp"]
    elif geom.name != "terrain":
      geom.contype = 0
      geom.conaffinity = 0

  key = spec.add_key()
  key.name = "init"
  key.qpos = [*INIT_STATE.pos, 1.0, 0.0, 0.0, 0.0, *default_joint_pos()]
  return spec


def get_go2_training_spec(timestep: float = SIM_TIMESTEP) -> mujoco.MjSpec:
  """Build the Go2 as the velocity task trains it.

  Ground plane, full collision set (every collision geom contacts the terrain:
  feet with contact-rich parameters, the rest as frictionless point contacts so
  non-foot ground touches are observable), position servos with armature, and
  the initial pose as a keyframe whose ctrl holds it.
  """
  spec = get_go2_spec()
  spec.option.timestep = timestep
  spec.option.iterations = SOLVER_ITERATIONS
  spec.option.ls_iterations = SOLVER_LS_ITERATIONS

  terrain = spec.worldbody.add_geom()
  terrain.name = "terrain"
  terrain.type = mujoco.mjtGeom.mjGEOM_PLANE
  terrain.size = [0, 0, 0.05]

  _apply_collision(spec, FULL_COLLISION)
  _add_position_actuators(spec)

  key = spec.add_key()
  key.name = "init"
  key.qpos = [*INIT_STATE.pos, 1.0, 0.0, 0.0, 0.0, *default_joint_pos()]
  key.ctrl = list(default_joint_pos())
  return spec
