"""Go2 flat-terrain velocity-tracking environment."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2
from unitree_rl_mjx.tasks.velocity.config.go2.env_cfgs import (
  unitree_go2_flat_env_cfg,
)
from unitree_rl_mjx.tasks.velocity.mdp import commands_vel, domain_randomize
from unitree_rl_mjx.tasks.velocity.mdp.types import MdpEnv
from unitree_rl_mjx.tasks.velocity.velocity_env_cfg import VelocityEnvCfg

_GRAVITY_DIR = jnp.array([0.0, 0.0, -1.0])
_CONTACT_DATASPEC = 3  # found | force
_VEL_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _wrap_to_pi(angle: jax.Array) -> jax.Array:
  return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _add_task_sensors(spec: mujoco.MjSpec, nonfoot_geoms: tuple[str, ...]) -> None:
  """Foot velocity and ground-contact sensors the task terms read.

  The gyro, velocimeter, and angular-momentum sensors ship with the robot
  description; contact sensing is per geom-vs-terrain pair (found + force,
  one slot), matching how the reference expands its contact sensor patterns
  into individual MuJoCo sensors.
  """
  for foot in go2.FOOT_NAMES:
    sensor = spec.add_sensor()
    sensor.name = f"{foot}_global_linvel"
    sensor.type = mujoco.mjtSensor.mjSENS_FRAMELINVEL
    sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
    sensor.objname = foot
  contact_pairs = [
    (f"{foot}_foot_ground", f"{foot}_foot_collision") for foot in go2.FOOT_NAMES
  ]
  contact_pairs += [(f"{geom}_ground_touch", geom) for geom in nonfoot_geoms]
  for name, geom in contact_pairs:
    sensor = spec.add_sensor()
    sensor.name = name
    sensor.type = mujoco.mjtSensor.mjSENS_CONTACT
    sensor.objtype = mujoco.mjtObj.mjOBJ_GEOM
    sensor.objname = geom
    sensor.reftype = mujoco.mjtObj.mjOBJ_GEOM
    sensor.refname = "terrain"
    sensor.intprm[0] = _CONTACT_DATASPEC
    sensor.intprm[1] = 0  # No reduction; a geom-plane pair has one slot.
    sensor.intprm[2] = 1


class Go2VelocityFlat(mjx_env.MjxEnv):
  """Velocity tracking on flat ground, mirroring the reference task config."""

  def __init__(
    self,
    cfg: VelocityEnvCfg | None = None,
    config_overrides: dict[str, Any] | None = None,
  ):
    self._cfg = cfg or unitree_go2_flat_env_cfg()
    ctrl_dt = self._cfg.sim.timestep * self._cfg.decimation
    super().__init__(
      config_dict.create(
        ctrl_dt=ctrl_dt,
        sim_dt=self._cfg.sim.timestep,
        episode_length=round(self._cfg.episode_length_s / ctrl_dt),
      ),
      config_overrides,
    )

    spec = go2.get_go2_training_spec(self._cfg.sim.timestep)
    spec.option.iterations = self._cfg.sim.iterations
    spec.option.ls_iterations = self._cfg.sim.ls_iterations
    _add_task_sensors(spec, go2.nonfoot_collision_geom_names())
    self._mj_model = spec.compile()
    self._mjx_model = mjx.put_model(self._mj_model)

    self._init_qpos = jnp.array(self._mj_model.key_qpos[0])
    self._default_joint_pos = jnp.array(go2.default_joint_pos())
    lowers, uppers = self._mj_model.jnt_range[1:].T
    mid = 0.5 * (lowers + uppers)
    span = 0.5 * (uppers - lowers) * go2.SOFT_JOINT_POS_LIMIT_FACTOR
    self._soft_joint_pos_lower = jnp.array(mid - span)
    self._soft_joint_pos_upper = jnp.array(mid + span)

    self._sensor_adr = {}
    for i in range(self._mj_model.nsensor):
      name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
      self._sensor_adr[name] = (
        self._mj_model.sensor_adr[i],
        self._mj_model.sensor_dim[i],
      )
    self._foot_site_ids = np.array(
      [self._mj_model.site(name).id for name in go2.FOOT_NAMES]
    )

    twist = self._cfg.commands["twist"]
    self._cmd_ranges = jnp.array(
      [twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z]
    )
    curriculum = self._cfg.curriculum.get("command_vel")
    self._velocity_stages = curriculum.params["velocity_stages"] if curriculum else ()

  ##
  # Properties
  ##

  @property
  def xml_path(self) -> str:
    return str(go2.GO2_XML)

  @property
  def action_size(self) -> int:
    return self._mj_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def domain_randomize_fn(self):
    """Model-field randomizer for the training wrapper, from the events cfg."""
    friction_range = self._cfg.events["foot_friction"].params["ranges"]
    com_ranges = self._cfg.events["base_com"].params["ranges"]
    foot_geom_ids = tuple(
      self._mj_model.geom(f"{name}_foot_collision").id for name in go2.FOOT_NAMES
    )
    base_body_id = self._mj_model.body("base_link").id
    return functools.partial(
      domain_randomize,
      foot_geom_ids=foot_geom_ids,
      base_body_id=base_body_id,
      friction_range=friction_range,
      com_offset_ranges=[com_ranges[i] for i in range(3)],
    )

  ##
  # Sensor access
  ##

  def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
    adr, dim = self._sensor_adr[name]
    return data.sensordata[adr : adr + dim]

  def _foot_contact_state(self, data: mjx.Data):
    """(found[4] bool, force[4, 3]) from the per-foot contact sensors."""
    readings = jnp.stack(
      [self._sensor(data, f"{foot}_foot_ground") for foot in go2.FOOT_NAMES]
    )
    return readings[:, 0] > 0, readings[:, 1:4]

  ##
  # Commands
  ##

  def _command_ranges(self, common_step_counter: jax.Array) -> jax.Array:
    if not self._velocity_stages:
      return self._cmd_ranges
    return commands_vel(common_step_counter, self._velocity_stages, self._cmd_ranges)

  def _resample_command(self, rng: jax.Array, common_step_counter: jax.Array):
    twist = self._cfg.commands["twist"]
    k_cmd, k_heading, k_stand, k_time = jax.random.split(rng, 4)
    ranges = self._command_ranges(common_step_counter)
    command = jax.random.uniform(k_cmd, (3,), minval=ranges[:, 0], maxval=ranges[:, 1])
    command = command * (jnp.linalg.norm(command) > 0.1)
    heading_target = jax.random.uniform(
      k_heading, minval=twist.ranges.heading[0], maxval=twist.ranges.heading[1]
    )
    is_standing = jax.random.uniform(k_stand) <= twist.rel_standing_envs
    steps_until_resample = jnp.round(
      jax.random.uniform(
        k_time,
        minval=twist.resampling_time_range[0],
        maxval=twist.resampling_time_range[1],
      )
      / self.dt
    ).astype(jnp.int32)
    return command, heading_target, is_standing, steps_until_resample

  def _update_command(self, info: dict[str, Any], data: mjx.Data) -> jax.Array:
    """Heading-servo the yaw rate and zero standing envs, every step."""
    twist = self._cfg.commands["twist"]
    command = info["command"]
    if twist.heading_command:
      forward = math.rotate(jnp.array([1.0, 0.0, 0.0]), data.qpos[3:7])
      heading_w = jnp.arctan2(forward[1], forward[0])
      heading_error = _wrap_to_pi(info["heading_target"] - heading_w)
      command = command.at[2].set(
        jnp.clip(
          twist.heading_control_stiffness * heading_error,
          twist.ranges.ang_vel_z[0],
          twist.ranges.ang_vel_z[1],
        )
      )
    return jnp.where(info["is_standing"], jnp.zeros(3), command)

  ##
  # MDP view
  ##

  def _mdp_env(
    self, data: mjx.Data, info: dict[str, Any], is_terminated: jax.Array
  ) -> MdpEnv:
    quat = data.qpos[3:7]
    foot_in_contact, foot_forces = self._foot_contact_state(data)
    sensors = {name: self._sensor(data, name) for name in self._sensor_adr}
    return MdpEnv(
      joint_names=go2.JOINT_ORDER,
      step_dt=self.dt,
      episode_length_buf=info["episode_length_buf"],
      command=info["command"],
      root_lin_vel_b=math.rotate(data.qvel[0:3], math.quat_inv(quat)),
      root_ang_vel_b=data.qvel[3:6],
      root_ang_vel_w=math.rotate(data.qvel[3:6], quat),
      projected_gravity_b=math.rotate(_GRAVITY_DIR, math.quat_inv(quat)),
      joint_pos=data.qpos[7:] + info["encoder_bias"],
      joint_vel=data.qvel[6:],
      joint_acc=data.qacc[6:],
      default_joint_pos=self._default_joint_pos,
      soft_joint_pos_lower=self._soft_joint_pos_lower,
      soft_joint_pos_upper=self._soft_joint_pos_upper,
      actions=info["actions"],
      last_actions=info["last_actions"],
      sensors=sensors,
      foot_in_contact=foot_in_contact,
      foot_forces=foot_forces,
      current_air_time=info["current_air_time"],
      current_contact_time=info["current_contact_time"],
      first_contact=info["first_contact"],
      foot_pos_w=data.site_xpos[self._foot_site_ids],
      foot_lin_vel_w=jnp.stack(
        [self._sensor(data, f"{f}_global_linvel") for f in go2.FOOT_NAMES]
      ),
      is_terminated=is_terminated,
    )

  ##
  # Observations
  ##

  def _get_obs(self, env: MdpEnv, info: dict[str, Any]) -> dict[str, jax.Array]:
    obs = {}
    group_keys = {"actor": "state", "critic": "privileged_state"}
    for group_name, group in self._cfg.observations.items():
      parts = []
      for term in group.terms.values():
        value = term.func(env, **term.params)
        if group.enable_corruption and term.noise is not None:
          info["rng"], key = jax.random.split(info["rng"])
          value = value + jax.random.uniform(
            key, value.shape, minval=term.noise.n_min, maxval=term.noise.n_max
          )
        parts.append(value)
      obs[group_keys[group_name]] = jnp.concatenate(parts)
    return obs

  ##
  # Reset / step
  ##

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_xy, k_yaw, k_cmd, k_push, k_bias = jax.random.split(rng, 6)

    pose_range = self._cfg.events["reset_base"].params["pose_range"]
    qpos = self._init_qpos
    dxy = jax.random.uniform(
      k_xy,
      (2,),
      minval=jnp.array([pose_range["x"][0], pose_range["y"][0]]),
      maxval=jnp.array([pose_range["x"][1], pose_range["y"][1]]),
    )
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    yaw = jax.random.uniform(
      k_yaw, minval=pose_range["yaw"][0], maxval=pose_range["yaw"][1]
    )
    yaw_quat = math.axis_angle_to_quat(jnp.array([0.0, 0.0, 1.0]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], yaw_quat))

    data = mjx.make_data(self._mj_model)
    data = data.replace(qpos=qpos, qvel=jnp.zeros(self._mjx_model.nv), ctrl=qpos[7:])
    data = mjx.forward(self._mjx_model, data)

    command, heading_target, is_standing, steps_until_resample = self._resample_command(
      k_cmd, jnp.zeros((), jnp.int32)
    )

    push_range = self._cfg.events["push_robot"].interval_range_s
    steps_until_push = jnp.round(
      jax.random.uniform(k_push, minval=push_range[0], maxval=push_range[1]) / self.dt
    ).astype(jnp.int32)

    info = {
      "rng": rng,
      "command": command,
      "heading_target": heading_target,
      "is_standing": is_standing,
      "steps_until_resample": steps_until_resample,
      "steps_until_push": steps_until_push,
      "actions": jnp.zeros(self._mj_model.nu),
      "last_actions": jnp.zeros(self._mj_model.nu),
      "current_air_time": jnp.zeros(4),
      "current_contact_time": jnp.zeros(4),
      "first_contact": jnp.zeros(4, dtype=bool),
      "episode_length_buf": jnp.zeros((), jnp.int32),
      # Encoder miscalibration, fixed for the env's lifetime: every joint_pos
      # read (observations and rewards alike) carries this bias, as in the
      # reference's entity data.
      "encoder_bias": jax.random.uniform(
        k_bias,
        (self._mj_model.nu,),
        minval=self._cfg.events["encoder_bias"].params["bias_range"][0],
        maxval=self._cfg.events["encoder_bias"].params["bias_range"][1],
      ),
    }
    metrics = {f"reward/{name}": jnp.zeros(()) for name in self._cfg.rewards}

    env = self._mdp_env(data, info, is_terminated=jnp.zeros((), bool))
    obs = self._get_obs(env, info)
    return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

  def _apply_push(self, info: dict[str, Any], data: mjx.Data) -> mjx.Data:
    """Interval push: add a sampled world-frame velocity delta to the root."""
    push = self._cfg.events["push_robot"]
    info["steps_until_push"] = info["steps_until_push"] - 1
    do_push = info["steps_until_push"] <= 0
    info["rng"], k_vel, k_time = jax.random.split(info["rng"], 3)
    ranges = jnp.array([push.params["velocity_range"][axis] for axis in _VEL_AXES])
    delta = jax.random.uniform(k_vel, (6,), minval=ranges[:, 0], maxval=ranges[:, 1])
    quat = data.qpos[3:7]
    # Free-joint qvel is world-frame linear, body-frame angular.
    delta = jnp.concatenate([delta[:3], math.rotate(delta[3:], math.quat_inv(quat))])
    qvel = data.qvel.at[0:6].set(
      jnp.where(do_push, data.qvel[0:6] + delta, data.qvel[0:6])
    )
    info["steps_until_push"] = jnp.where(
      do_push,
      jnp.round(
        jax.random.uniform(
          k_time,
          minval=push.interval_range_s[0],
          maxval=push.interval_range_s[1],
        )
        / self.dt
      ).astype(jnp.int32),
      info["steps_until_push"],
    )
    return data.replace(qvel=qvel)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    data = self._apply_push(info, state.data)

    action_cfg = self._cfg.actions["joint_pos"]
    motor_targets = self._default_joint_pos + action * action_cfg.scale
    data = mjx_env.step(self._mjx_model, data, motor_targets, self.n_substeps)

    # Contact bookkeeping at control rate, mirroring the reference's air-time
    # state: first contact is judged against the pre-update air time.
    foot_in_contact, _ = self._foot_contact_state(data)
    info["first_contact"] = (info["current_air_time"] > 0.0) & foot_in_contact
    info["current_air_time"] = jnp.where(
      ~foot_in_contact, info["current_air_time"] + self.dt, 0.0
    )
    info["current_contact_time"] = jnp.where(
      foot_in_contact, info["current_contact_time"] + self.dt, 0.0
    )

    info["episode_length_buf"] = info["episode_length_buf"] + 1
    info["last_actions"] = info["actions"]
    info["actions"] = action

    # Command resampling and per-step heading/standing updates.
    info["steps_until_resample"] = info["steps_until_resample"] - 1
    info["rng"], k_resample = jax.random.split(info["rng"])
    resampled = self._resample_command(k_resample, info["episode_length_buf"])
    do_resample = info["steps_until_resample"] <= 0
    for key, value in zip(
      ("command", "heading_target", "is_standing", "steps_until_resample"),
      resampled,
    ):
      info[key] = jnp.where(do_resample, value, info[key])
    info["command"] = self._update_command(info, data)

    env = self._mdp_env(data, info, is_terminated=jnp.zeros((), bool))
    done = jnp.zeros((), bool)
    for term in self._cfg.terminations.values():
      if term.time_out:
        continue
      done = done | term.func(env, **term.params)
    env = env.replace(is_terminated=done)

    reward = jnp.zeros(())
    for name, term in self._cfg.rewards.items():
      value = term.func(env, **term.params)
      state.metrics[f"reward/{name}"] = value * term.weight
      reward = reward + value * term.weight
    reward = reward * self.dt

    obs = self._get_obs(env, info)
    return mjx_env.State(
      data, obs, reward, done.astype(jnp.float32), state.metrics, info
    )
