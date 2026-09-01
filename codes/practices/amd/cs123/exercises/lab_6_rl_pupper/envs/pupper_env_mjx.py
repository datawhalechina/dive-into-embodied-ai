"""Pupper RL 的 MJX/brax 训练环境。

与 CPU 版 `pupper_env.py` 平行：CPU 版给 SB3 用，本文件给 brax-PPO 用。
观测 36 维（× 15 帧历史栈 = 540 维），动作 12 维，18 项奖励，包含 kick + 观测
噪声 + 初始位姿等域随机化。命令分布 `vx ±0.75 / vy ±0.5 / wz ±2.0`。

训练验证：200M 步 brax-PPO（num_envs=8192, batch_size=256），在约 149M 步处
达到峰值 ep_rew ≈ 51.15 ± 7.4，存活率 100%。
"""

from __future__ import annotations

import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import mujoco
import numpy as np
from brax import base, math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp

EXERCISES_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = str(EXERCISES_DIR / "shared" / "models" / "pupper_v3_floating.xml")


# ----------------------- 域随机化辅助 -----------------------


@dataclass
class StartPositionRandomization:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.15
    z_max: float = 0.20


def _random_z_rotation_quat(rng):
    yaw = jax.random.uniform(rng, (1,), minval=-jp.pi, maxval=jp.pi)
    return jp.concatenate((jp.cos(yaw / 2), jp.zeros(2), jp.sin(yaw / 2)))


def _randomize_qpos(qpos: jax.Array, cfg: StartPositionRandomization, rng) -> jax.Array:
    key_pos, key_yaw = jax.random.split(rng, 2)
    qpos = qpos.at[:3].set(
        jax.random.uniform(
            key_pos,
            shape=(3,),
            minval=jp.array((cfg.x_min, cfg.y_min, cfg.z_min)),
            maxval=jp.array((cfg.x_max, cfg.y_max, cfg.z_max)),
        )
    )
    qpos = qpos.at[3:7].set(_random_z_rotation_quat(key_yaw))
    return qpos


def _push_front(buffer: jax.Array, new_value: jax.Array) -> jax.Array:
    buffer = jp.roll(buffer, shift=1, axis=1)
    return buffer.at[:, 0].set(new_value)


def _sample_lagged_value(rng, buffer_newest_first, new_value, distribution):
    buf = _push_front(buffer_newest_first, new_value)
    return jax.random.choice(rng, buf, axis=1, p=distribution), buf


# ----------------------- reward 函数 -----------------------


def _r_lin_vel_z(xd: Motion):
    return jp.square(xd.vel[0, 2])


def _r_ang_vel_xy(xd: Motion):
    return jp.sum(jp.square(xd.ang[0, :2]))


def _r_tracking_orientation(desired_world_z_in_body_frame, x: Transform, sigma):
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    err = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    return jp.exp(-err / sigma)


def _r_orientation(x: Transform):
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))


def _r_torques(torques):
    return jp.sum(jp.square(torques))


def _r_joint_acc(joint_vel, last_joint_vel, dt):
    return jp.sum(jp.square((joint_vel - last_joint_vel) / dt))


def _r_action_rate(act, last_act):
    return jp.sum(jp.square(act - last_act))


def _r_tracking_lin_vel(commands, x: Transform, xd: Motion, sigma):
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    err = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-err / sigma)


def _r_tracking_ang_vel(commands, x: Transform, xd: Motion, sigma):
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    err = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-err / sigma)


def _r_feet_air_time(air_time, first_contact, commands, minimum_airtime=0.1):
    r = jp.sum((air_time - minimum_airtime) * first_contact)
    r *= math.normalize(commands[:3])[1] > 0.05
    return r


def _r_abduction_angle(joint_angles, desired):
    return jp.sum(jp.square(joint_angles[1::3] - desired))


def _r_stand_still(commands, values, default, threshold):
    return jp.sum(jp.abs(values - default)) * (math.normalize(commands[:3])[1] < threshold)


def _r_foot_slip(pipeline_state, contact_filt, feet_site_id, lower_leg_body_id):
    pos = pipeline_state.site_xpos[feet_site_id]
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))


def _r_termination(done, step, step_threshold):
    return done & (step < step_threshold)


def _r_geom_collision(pipeline_state, geom_ids):
    contact = jp.array(0.0)
    for gid in geom_ids:
        contact += jp.sum(
            ((pipeline_state.contact.geom1 == gid) | (pipeline_state.contact.geom2 == gid))
            * (pipeline_state.contact.dist < 0.0)
        )
    return contact


# ----------------------- 默认 reward 配置 -----------------------


def default_reward_config() -> Dict[str, Any]:
    """默认 18 项奖励权重表。

    注意权重符号：penalty 项写负、tracking 项写正。最终 reward = sum(scale * term) * dt。
    """
    return {
        "scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            "tracking_orientation": 1.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -5.0,
            "torques": -2e-4,
            "joint_acceleration": -1e-6,
            "mechanical_work": 0.0,
            "action_rate": -0.01,
            "feet_air_time": 0.2,
            "stand_still": -0.5,
            "stand_still_joint_velocity": -0.1,
            "abduction_angle": -0.1,
            "termination": -100.0,
            "foot_slip": -0.1,
            "knee_collision": -1.0,
            "body_collision": -1.0,
        },
        "tracking_sigma": 0.25,
    }


# ----------------------- 主环境类 -----------------------


def _body_name_to_geom_ids(mj_model, body_name: str) -> np.ndarray:
    body = mj_model.body(body_name)
    return body.geomadr + np.arange(int(np.squeeze(body.geomnum)))


def _body_names_to_geom_ids(mj_model, names: Sequence[str]) -> np.ndarray:
    return np.concatenate([_body_name_to_geom_ids(mj_model, n) for n in names])


def _body_names_to_body_ids(mj_model, names: Sequence[str]) -> np.ndarray:
    ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, n) for n in names]
    assert not any(i == -1 for i in ids), f"Body not found among {names}"
    return np.array(ids)


class PupperV3MJXEnv(PipelineEnv):
    """Pupper V3 在 MJX/brax 下的训练环境。"""

    def __init__(
        self,
        path: str = DEFAULT_XML,
        reward_config: Optional[Dict[str, Any]] = None,
        action_scale: float = 1.0,
        observation_history: int = 15,
        joint_lower_limits: Tuple[float, ...] = (
            -1.220, -0.420, -2.790, -2.510, -3.140, -0.710,
            -1.220, -0.420, -2.790, -2.510, -3.140, -0.710,
        ),
        joint_upper_limits: Tuple[float, ...] = (
            2.510, 3.140, 0.710, 1.220, 0.420, 2.790,
            2.510, 3.140, 0.710, 1.220, 0.420, 2.790,
        ),
        dof_damping: float = 0.25,
        position_control_kp: float = 5.0,
        start_position_config: Optional[StartPositionRandomization] = None,
        foot_site_names: Tuple[str, ...] = (
            "leg_front_r_3_foot_site",
            "leg_front_l_3_foot_site",
            "leg_back_r_3_foot_site",
            "leg_back_l_3_foot_site",
        ),
        torso_name: str = "base_link",
        upper_leg_body_names: Tuple[str, ...] = (
            "leg_front_r_2", "leg_front_l_2", "leg_back_r_2", "leg_back_l_2",
        ),
        lower_leg_body_names: Tuple[str, ...] = (
            "leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3",
        ),
        resample_velocity_step: int = 500,
        linear_velocity_x_range: Tuple[float, float] = (-0.75, 0.75),
        linear_velocity_y_range: Tuple[float, float] = (-0.5, 0.5),
        angular_velocity_range: Tuple[float, float] = (-2.0, 2.0),
        zero_command_probability: float = 0.01,
        stand_still_command_threshold: float = 0.1,
        maximum_pitch_command: float = 0.0,
        maximum_roll_command: float = 0.0,
        default_pose: Tuple[float, ...] = (
            0.26, 0.0, -0.52, -0.26, 0.0, 0.52,
            0.26, 0.0, -0.52, -0.26, 0.0, 0.52,
        ),
        desired_abduction_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        angular_velocity_noise: float = 0.3,
        gravity_noise: float = 0.1,
        motor_angle_noise: float = 0.1,
        last_action_noise: float = 0.01,
        kick_vel: float = 0.2,
        kick_probability: float = 0.02,
        terminal_body_z: float = 0.1,
        early_termination_step_threshold: int = 500,
        terminal_body_angle: float = 0.52,
        foot_radius: float = 0.02,
        environment_timestep: float = 0.02,
        physics_timestep: float = 0.004,
        latency_distribution: Tuple[float, ...] = (0.2, 0.8),
        imu_latency_distribution: Tuple[float, ...] = (0.5, 0.5),
        desired_world_z_in_body_frame: Tuple[float, ...] = (0.0, 0.0, 1.0),
        use_imu: bool = True,
    ):
        if start_position_config is None:
            start_position_config = StartPositionRandomization()
        if reward_config is None:
            reward_config = default_reward_config()

        default_pose_arr = jp.array(default_pose)
        desired_abduction_arr = jp.array(desired_abduction_angles)
        latency_dist_arr = jp.array(latency_distribution)
        imu_latency_dist_arr = jp.array(imu_latency_distribution)

        sys_model = mjcf.load(path)
        self._dt = environment_timestep
        sys_model = sys_model.tree_replace({"opt.timestep": physics_timestep})
        sys_model = sys_model.replace(
            actuator_gainprm=sys_model.actuator_gainprm.at[:, 0].set(position_control_kp),
            actuator_biasprm=sys_model.actuator_biasprm.at[:, 1].set(-position_control_kp).at[:, 2].set(-dof_damping),
        )
        sys_model.mj_model.keyframe("home").qpos[7:] = np.array(default_pose)

        n_frames = int(self._dt // sys_model.opt.timestep)
        super().__init__(sys_model, backend="mjx", n_frames=n_frames)

        self._reward_config = reward_config
        self._scales = reward_config["scales"]
        self._tracking_sigma = reward_config["tracking_sigma"]

        self._torso_geom_ids = _body_name_to_geom_ids(sys_model.mj_model, torso_name)
        self._torso_idx = mujoco.mj_name2id(sys_model.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, torso_name)
        assert self._torso_idx != -1

        self._action_scale = jp.array(action_scale)
        self._angular_velocity_noise = angular_velocity_noise
        self._gravity_noise = gravity_noise
        self._motor_angle_noise = motor_angle_noise
        self._last_action_noise = last_action_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys_model.mj_model.keyframe("home").qpos)
        self._default_pose = default_pose_arr
        self._desired_abduction_angles = desired_abduction_arr
        self.lowers = jp.array(joint_lower_limits)
        self.uppers = jp.array(joint_upper_limits)

        feet_site_id = [
            mujoco.mj_name2id(sys_model.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in foot_site_names
        ]
        assert not any(i == -1 for i in feet_site_id), "foot site missing"
        self._feet_site_id = np.array(feet_site_id)

        self._lower_leg_body_id = _body_names_to_body_ids(sys_model.mj_model, lower_leg_body_names)
        self._upper_leg_geom_ids = _body_names_to_geom_ids(sys_model.mj_model, upper_leg_body_names)

        self._foot_radius = foot_radius
        self._nv = sys_model.nv

        self._start_position_config = start_position_config
        self._linear_velocity_x_range = linear_velocity_x_range
        self._linear_velocity_y_range = linear_velocity_y_range
        self._angular_velocity_range = angular_velocity_range
        self._zero_command_probability = zero_command_probability
        self._stand_still_command_threshold = stand_still_command_threshold
        self._maximum_pitch_command = maximum_pitch_command
        self._maximum_roll_command = maximum_roll_command
        self._kick_probability = kick_probability
        self._resample_velocity_step = resample_velocity_step

        self.observation_dim = 36
        self._observation_history = observation_history

        self._early_termination_step_threshold = early_termination_step_threshold
        self._terminal_body_z = terminal_body_z
        self._terminal_body_angle = terminal_body_angle
        self._desired_world_z_in_body_frame = jp.array(desired_world_z_in_body_frame)
        self._latency_distribution = latency_dist_arr
        self._imu_latency_distribution = imu_latency_dist_arr
        self._use_imu = use_imu

    # ------------- sampling helpers -------------

    def sample_command(self, rng: jax.Array) -> jax.Array:
        key1, key2, key3, key4, key5 = jax.random.split(rng, 5)
        vx = jax.random.uniform(key1, (1,), minval=self._linear_velocity_x_range[0], maxval=self._linear_velocity_x_range[1])
        vy = jax.random.uniform(key2, (1,), minval=self._linear_velocity_y_range[0], maxval=self._linear_velocity_y_range[1])
        wz = jax.random.uniform(key3, (1,), minval=self._angular_velocity_range[0], maxval=self._angular_velocity_range[1])
        new_cmd = jp.array([vx[0], vy[0], wz[0]])

        zero_prob = jax.random.uniform(key4, (1,))
        near_zero = jax.random.uniform(
            key5, (3,),
            minval=-self._stand_still_command_threshold,
            maxval=self._stand_still_command_threshold,
        )
        return jp.where(zero_prob < self._zero_command_probability, near_zero, new_cmd)

    def sample_body_orientation(self, rng: jax.Array) -> jax.Array:
        key_pitch, key_roll = jax.random.split(rng, 2)
        pitch = jax.random.uniform(key_pitch, (1,), minval=-1.0, maxval=1.0) * self._maximum_pitch_command
        roll = jax.random.uniform(key_roll, (1,), minval=-1.0, maxval=1.0) * self._maximum_roll_command
        euler = math.euler_to_quat(jp.array([roll[0], pitch[0], 0.0]))
        return math.rotate(self._desired_world_z_in_body_frame, euler)

    def initial_action_buffer(self) -> jax.Array:
        return jp.zeros((12, self._latency_distribution.shape[0]), dtype=float)

    def initial_imu_buffer(self) -> jax.Array:
        buf = jp.zeros((6, self._imu_latency_distribution.shape[0]), dtype=float)
        return buf.at[5, :].set(-1.0)

    # ------------- reset / step -------------

    def reset(self, rng: jax.Array) -> State:
        rng, cmd_key, orient_key, pos_key = jax.random.split(rng, 4)
        init_q = _randomize_qpos(self._init_q, self._start_position_config, pos_key)
        pipeline_state = self.pipeline_init(init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12, dtype=float),
            "action_buffer": self.initial_action_buffer(),
            "imu_buffer": self.initial_imu_buffer(),
            "last_vel": jp.zeros(12, dtype=float),
            "command": self.sample_command(cmd_key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4, dtype=float),
            "rewards": {k: 0.0 for k in self._scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "desired_world_z_in_body_frame": self.sample_body_orientation(orient_key),
        }

        obs_history = jp.zeros(self._observation_history * self.observation_dim, dtype=float)
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2, dtype=float)
        metrics = {"total_dist": 0.0}
        for k, v in state_info["rewards"].items():
            metrics[k] = v
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        state.info["rng"], cmd_rng, kick_dir_key, kick_bern_key, latency_key = jax.random.split(
            state.info["rng"], 5
        )

        # random kicks
        kick = jax.random.uniform(kick_dir_key, shape=(2,), minval=-1.0, maxval=1.0) * self._kick_vel
        kick *= jax.random.bernoulli(kick_bern_key, p=self._kick_probability, shape=(1,))
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # action latency
        lagged_action, state.info["action_buffer"] = _sample_lagged_value(
            latency_key, state.info["action_buffer"], action, self._latency_distribution
        )

        # physics step
        motor_targets = self._default_pose + lagged_action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done flags
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < np.cos(self._terminal_body_angle)
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._terminal_body_z

        # reward terms
        rewards_dict = {
            "tracking_lin_vel": _r_tracking_lin_vel(state.info["command"], x, xd, self._tracking_sigma),
            "tracking_ang_vel": _r_tracking_ang_vel(state.info["command"], x, xd, self._tracking_sigma),
            "tracking_orientation": _r_tracking_orientation(
                state.info["desired_world_z_in_body_frame"], x, self._tracking_sigma
            ),
            "lin_vel_z": _r_lin_vel_z(xd),
            "ang_vel_xy": _r_ang_vel_xy(xd),
            "orientation": _r_orientation(x),
            "torques": _r_torques(pipeline_state.qfrc_actuator),
            "joint_acceleration": _r_joint_acc(joint_vel, state.info["last_vel"], dt=self._dt),
            "mechanical_work": jp.sum(jp.abs(pipeline_state.qfrc_actuator[6:] * pipeline_state.qvel[6:])),
            "action_rate": _r_action_rate(action, state.info["last_act"]),
            "stand_still": _r_stand_still(state.info["command"], joint_angles, self._default_pose, 0.1),
            "stand_still_joint_velocity": _r_stand_still(
                state.info["command"], joint_vel, jp.zeros(12), self._stand_still_command_threshold
            ),
            "abduction_angle": _r_abduction_angle(joint_angles, self._desired_abduction_angles),
            "feet_air_time": _r_feet_air_time(state.info["feet_air_time"], first_contact, state.info["command"]),
            "foot_slip": _r_foot_slip(pipeline_state, contact_filt_cm, self._feet_site_id, self._lower_leg_body_id),
            "termination": _r_termination(done, state.info["step"], self._early_termination_step_threshold),
            "knee_collision": _r_geom_collision(pipeline_state, self._upper_leg_geom_ids),
            "body_collision": _r_geom_collision(pipeline_state, self._torso_geom_ids),
        }
        rewards_dict = {k: v * self._scales[k] for k, v in rewards_dict.items()}
        reward = jp.clip(sum(rewards_dict.values()) * self.dt, 0.0, 10000.0)

        # state info bookkeeping
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards_dict
        state.info["step"] += 1

        # command / orientation resample
        state.info["command"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["desired_world_z_in_body_frame"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_body_orientation(cmd_rng),
            state.info["desired_world_z_in_body_frame"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._resample_velocity_step),
            0,
            state.info["step"],
        )

        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    # ------------- observation -------------

    def _get_obs(self, pipeline_state: base.State, state_info: Dict[str, Any], obs_history: jax.Array) -> jax.Array:
        if self._use_imu:
            inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
            local_ang_vel = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        else:
            inv_torso_rot = jp.array([1.0, 0.0, 0.0, 0.0])
            local_ang_vel = jp.zeros(3)

        (
            state_info["rng"],
            ang_key, grav_key, motor_key, last_act_key, imu_key,
        ) = jax.random.split(state_info["rng"], 6)

        ang_noise = jax.random.uniform(ang_key, (3,), minval=-1, maxval=1) * self._angular_velocity_noise
        grav_noise = jax.random.uniform(grav_key, (3,), minval=-1, maxval=1) * self._gravity_noise
        motor_noise = jax.random.uniform(motor_key, (12,), minval=-1, maxval=1) * self._motor_angle_noise
        last_act_noise = jax.random.uniform(last_act_key, (12,), minval=-1, maxval=1) * self._last_action_noise

        noised_gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_torso_rot) + grav_noise
        noised_gravity = noised_gravity / jp.linalg.norm(noised_gravity)
        noised_ang_vel = local_ang_vel + ang_noise
        noised_imu = jp.concatenate([noised_ang_vel, noised_gravity])

        lagged_imu, state_info["imu_buffer"] = _sample_lagged_value(
            imu_key, state_info["imu_buffer"], noised_imu, self._imu_latency_distribution
        )

        obs = jp.concatenate([
            lagged_imu,
            state_info["command"],
            state_info["desired_world_z_in_body_frame"],
            pipeline_state.q[7:] - self._default_pose + motor_noise,
            state_info["last_act"] + last_act_noise,
        ])
        assert self.observation_dim == obs.shape[0]
        obs = jp.clip(obs, -100.0, 100.0)
        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def render(self, trajectory, camera: Optional[str] = None, **kwargs):
        return super().render(trajectory, camera=camera or "track", **kwargs)
