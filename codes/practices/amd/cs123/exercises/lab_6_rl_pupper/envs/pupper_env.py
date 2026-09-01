"""Pupper RL Gymnasium 环境（CPU/SB3 路径）。

奖励配方 + 命令分布 + 观测设计 + 域随机化全部与 MJX/brax 版（`pupper_env_mjx.py`）
对齐，使 SB3 / SubprocVecEnv 路径能复现 brax-PPO 200M 步的训练结果。

关键设计：
- 观测：36 维 / step × 15 帧 stack = 540（IMU 6 + cmd 3 + desired_world_z 3 + 关节角差 12
  + last_act 12），含 IMU latency + 观测噪声 + 动作 latency。
- 奖励：18 项，penalty 项 term 返回正幅值、权重写负；tracking 项权重写正。
  最终 reward = sum(scale × term) × dt，clip 到 [0, 10000]。
- 命令：vx ∈ [-0.75, 0.75]，vy ∈ [-0.5, 0.5]，wz ∈ [-2.0, 2.0]，
  以 1% 概率回零命令，step 500 处中途重采样。
- 域随机化：随机初始位姿 + 偶发 kick + 观测噪声 + 动作延迟。
- 物理：250 Hz physics（0.004 s）× 5 = 50 Hz control（0.02 s），与 MJX 版一致。
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

EXERCISES_DIR = Path(__file__).resolve().parents[2]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.rl.obs_helpers import base_local_gravity  # noqa: E402

DEFAULT_XML = str(EXERCISES_DIR / "shared" / "models" / "pupper_v3_floating.xml")

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1",  "leg_back_r_2",  "leg_back_r_3",
    "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
]

FOOT_BODIES = [
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
]

FOOT_SITES = [
    "leg_front_r_3_foot_site",
    "leg_front_l_3_foot_site",
    "leg_back_r_3_foot_site",
    "leg_back_l_3_foot_site",
]

UPPER_LEG_BODIES = [
    "leg_front_r_2",
    "leg_front_l_2",
    "leg_back_r_2",
    "leg_back_l_2",
]

# Reward weight table — penalty 项 term 返回正幅值，权重写负；tracking 项写正。
REWARD_WEIGHTS = {
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
}

TRACKING_SIGMA = 0.25
AIR_TIME_TARGET = 0.1
CMD_DEAD_ZONE = 0.05
STAND_STILL_THRESHOLD = 0.1

KP = 5.0
KD = 0.25
ACTION_SCALE = 1.0
MAX_STEPS = 1000

# Bar F 物理：control 50 Hz，physics 250 Hz
DT_CONTROL = 0.02
DT_PHYSICS = 0.004

# 命令分布
CMD_VX_RANGE = (-0.75, 0.75)
CMD_VY_RANGE = (-0.5, 0.5)
CMD_WZ_RANGE = (-2.0, 2.0)
ZERO_CMD_PROB = 0.01
RESAMPLE_STEP = 500

# default_pose（与 MJX 版 literal 一致；不再用 IK 求 stance_q）
DEFAULT_POSE = np.array(
    [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52],
    dtype=np.float64,
)
DESIRED_ABDUCTION = np.zeros(4, dtype=np.float64)

# 终止条件
TERMINAL_BODY_Z = 0.10
TERMINAL_BODY_ANGLE = 0.52  # rad ≈ 30°
EARLY_TERMINATION_STEP = 500

# 域随机化幅度
ANG_VEL_NOISE = 0.3
GRAVITY_NOISE = 0.1
MOTOR_ANGLE_NOISE = 0.1
LAST_ACTION_NOISE = 0.01
KICK_VEL = 0.2
KICK_PROBABILITY = 0.02
START_POS_RANGE = (-2.0, 2.0, -2.0, 2.0, 0.15, 0.20)  # x_min, x_max, y_min, y_max, z_min, z_max

# action / IMU latency 分布（与 MJX 版一致）
ACTION_LATENCY_DIST = np.array([0.2, 0.8], dtype=np.float64)  # 80% lagged by 1 step
IMU_LATENCY_DIST = np.array([0.5, 0.5], dtype=np.float64)

# 关节角软限位（与 MJX 版一致）
JOINT_LOWERS = np.array(
    [-1.220, -0.420, -2.790, -2.510, -3.140, -0.710,
     -1.220, -0.420, -2.790, -2.510, -3.140, -0.710],
    dtype=np.float64,
)
JOINT_UPPERS = np.array(
    [2.510, 3.140, 0.710, 1.220, 0.420, 2.790,
     2.510, 3.140, 0.710, 1.220, 0.420, 2.790],
    dtype=np.float64,
)

# Frame stack 大小，与 MJX 版的 observation_history 一致
N_STACK = 15
OBS_DIM_PER_STEP = 36
OBS_DIM_STACKED = N_STACK * OBS_DIM_PER_STEP  # = 540

FOOT_RADIUS = 0.02


class PupperEnv(gym.Env):
    """CPU/SB3 训练用 Pupper 环境，与 MJX/brax 版逐项对齐。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml: str = DEFAULT_XML, max_steps: int = MAX_STEPS):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT_PHYSICS
        self.dt = DT_CONTROL
        self.n_substeps = max(1, int(round(DT_CONTROL / DT_PHYSICS)))
        self.max_steps = max_steps

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(
            -100.0, 100.0, (OBS_DIM_STACKED,), np.float32,
        )

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._joint_qpos_ids = []
        self._joint_qvel_ids = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            self._joint_qvel_ids.append(int(self.model.jnt_dofadr[jid]))
        self._joint_qpos_ids = np.array(self._joint_qpos_ids, dtype=int)
        self._joint_qvel_ids = np.array(self._joint_qvel_ids, dtype=int)

        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES
        ], dtype=int)
        self._lower_leg_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES
        ], dtype=int)

        # geom ids for collision-based penalties
        body_id_torso = self._base_id
        self._torso_geom_ids = self._body_to_geom_ids(body_id_torso)
        self._upper_leg_geom_ids = np.concatenate([
            self._body_to_geom_ids(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n))
            for n in UPPER_LEG_BODIES
        ])

        self._setup_pd()

        # 内部 buffer
        self._action_buffer = np.zeros((12, ACTION_LATENCY_DIST.shape[0]), dtype=np.float64)
        self._imu_buffer = np.zeros((6, IMU_LATENCY_DIST.shape[0]), dtype=np.float64)
        self._imu_buffer[5, :] = -1.0  # gravity z=-1 initially
        self._obs_history = np.zeros(OBS_DIM_STACKED, dtype=np.float32)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_joint_vel = np.zeros(12, dtype=np.float64)
        self.last_contact = np.zeros(4, dtype=bool)
        self.feet_air_time = np.zeros(4, dtype=np.float64)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.desired_world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.step_count = 0

    def _body_to_geom_ids(self, body_id: int) -> np.ndarray:
        adr = int(self.model.body_geomadr[body_id])
        n = int(self.model.body_geomnum[body_id])
        return np.arange(adr, adr + n, dtype=int)

    def _setup_pd(self) -> None:
        self.model.actuator_gainprm[:, 0] = KP
        self.model.actuator_biasprm[:, 1] = -KP
        self.model.actuator_biasprm[:, 2] = -KD

    # ----------------- gym API -----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 随机初始位置 + yaw
        x_min, x_max, y_min, y_max, z_min, z_max = START_POS_RANGE
        x0 = self.np_random.uniform(x_min, x_max)
        y0 = self.np_random.uniform(y_min, y_max)
        z0 = self.np_random.uniform(z_min, z_max)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0] = x0
        self.data.qpos[1] = y0
        self.data.qpos[2] = z0
        self.data.qpos[3] = np.cos(yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(yaw / 2)
        self.data.qpos[self._joint_qpos_ids] = DEFAULT_POSE
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)

        self._action_buffer[:] = 0.0
        self._imu_buffer[:] = 0.0
        self._imu_buffer[5, :] = -1.0
        self._obs_history[:] = 0.0

        self.last_action[:] = 0.0
        self.last_joint_vel[:] = 0.0
        self.last_contact[:] = False
        self.feet_air_time[:] = 0.0
        self.cmd = self._sample_command()
        self.desired_world_z[:] = np.array([0.0, 0.0, 1.0])
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # 偶发 kick：base 线速度被打扰
        if self.np_random.uniform() < KICK_PROBABILITY:
            dvx = self.np_random.uniform(-1, 1) * KICK_VEL
            dvy = self.np_random.uniform(-1, 1) * KICK_VEL
            self.data.qvel[0] += dvx
            self.data.qvel[1] += dvy

        # action latency
        lagged_action, self._action_buffer = _sample_lagged(
            self.np_random, self._action_buffer, action.astype(np.float64), ACTION_LATENCY_DIST,
        )
        motor_targets = DEFAULT_POSE + lagged_action * ACTION_SCALE
        motor_targets = np.clip(motor_targets, JOINT_LOWERS, JOINT_UPPERS)
        self.data.ctrl[:] = motor_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        joint_angles = self.data.qpos[self._joint_qpos_ids].astype(np.float64)
        joint_vel = self.data.qvel[self._joint_qvel_ids].astype(np.float64)

        # foot contact via site z
        foot_z = self.data.site_xpos[self._foot_site_ids][:, 2] - FOOT_RADIUS
        contact = foot_z < 1e-3
        contact_filt_mm = contact | self.last_contact
        contact_filt_cm = (foot_z < 3e-2) | self.last_contact
        first_contact = (self.feet_air_time > 0) & contact_filt_mm
        self.feet_air_time = self.feet_air_time + self.dt

        # 终止判定
        up_in_body = self._gravity_in_body() * -1.0  # gravity 是 (-z) in world，body 帧投影后 z 分量
        done_orient = up_in_body[2] < np.cos(TERMINAL_BODY_ANGLE)
        done_joint = bool(np.any(joint_angles < JOINT_LOWERS) or np.any(joint_angles > JOINT_UPPERS))
        done_height = self.data.qpos[2] < TERMINAL_BODY_Z
        terminated = bool(done_orient or done_joint or done_height)

        reward, info = self._compute_reward(action, joint_angles, joint_vel, contact_filt_cm, first_contact, terminated)

        # 更新 state.info
        self.last_action = action.copy()
        self.last_joint_vel = joint_vel
        self.feet_air_time = self.feet_air_time * (~contact_filt_mm).astype(np.float64)
        self.last_contact = contact
        self.step_count += 1

        # cmd resample at step 500
        if self.step_count > RESAMPLE_STEP:
            self.cmd = self._sample_command()
            self.step_count = 0

        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ----------------- reward -----------------

    def _compute_reward(self, action, joint_angles, joint_vel, contact_filt_cm, first_contact, terminated):
        base_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        # Body-frame velocities
        R_world_body = self.data.xmat[self._base_id].reshape(3, 3)
        local_lin = R_world_body.T @ base_vel
        local_ang = R_world_body.T @ base_ang_vel

        vx_cmd, vy_cmd, wz_cmd = float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2])
        cmd_mag = float(np.linalg.norm(self.cmd))

        # Tracking rewards (positive returns)
        lin_err = (local_lin[0] - vx_cmd) ** 2 + (local_lin[1] - vy_cmd) ** 2
        r_tracking_lin_vel = float(np.exp(-lin_err / TRACKING_SIGMA))
        ang_err = (local_ang[2] - wz_cmd) ** 2
        r_tracking_ang_vel = float(np.exp(-ang_err / TRACKING_SIGMA))

        world_z_in_body = R_world_body.T @ np.array([0.0, 0.0, 1.0])
        orient_err = float(np.sum((world_z_in_body - self.desired_world_z) ** 2))
        r_tracking_orientation = float(np.exp(-orient_err / TRACKING_SIGMA))

        # Penalty terms (positive magnitudes; weights carry the sign)
        r_lin_vel_z = float(local_lin[2] ** 2)
        r_ang_vel_xy = float(local_ang[0] ** 2 + local_ang[1] ** 2)
        rot_up = R_world_body @ np.array([0.0, 0.0, 1.0])
        r_orientation = float(rot_up[0] ** 2 + rot_up[1] ** 2)

        torques = self.data.qfrc_actuator[self._joint_qvel_ids]
        r_torques = float(np.sum(torques ** 2))
        r_joint_acceleration = float(np.sum(((joint_vel - self.last_joint_vel) / self.dt) ** 2))
        r_mechanical_work = float(np.sum(np.abs(torques * joint_vel)))
        r_action_rate = float(np.sum((action.astype(np.float64) - self.last_action.astype(np.float64)) ** 2))

        r_abduction_angle = float(np.sum((joint_angles[1::3] - DESIRED_ABDUCTION) ** 2))

        if cmd_mag < STAND_STILL_THRESHOLD:
            r_stand_still = float(np.sum(np.abs(joint_angles - DEFAULT_POSE)))
            r_stand_still_joint_velocity = float(np.sum(np.abs(joint_vel)))
        else:
            r_stand_still = 0.0
            r_stand_still_joint_velocity = 0.0

        # feet_air_time（cmd 大于 dead zone 才有奖励）
        if cmd_mag > CMD_DEAD_ZONE:
            r_feet_air_time = float(np.sum((self.feet_air_time - AIR_TIME_TARGET) * first_contact))
        else:
            r_feet_air_time = 0.0

        # foot slip
        slip_sq = 0.0
        for i, body_id in enumerate(self._lower_leg_body_ids):
            if contact_filt_cm[i]:
                v = self.data.cvel[body_id, 3:5]
                slip_sq += float(v[0] ** 2 + v[1] ** 2)
        r_foot_slip = slip_sq

        # collisions
        n_body, n_knee = 0, 0
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in self._torso_geom_ids or g2 in self._torso_geom_ids:
                n_body += 1
            if g1 in self._upper_leg_geom_ids or g2 in self._upper_leg_geom_ids:
                n_knee += 1
        r_body_collision = float(n_body)
        r_knee_collision = float(n_knee)

        # termination
        r_termination = 1.0 if (terminated and self.step_count < EARLY_TERMINATION_STEP) else 0.0

        terms = {
            "tracking_lin_vel": r_tracking_lin_vel,
            "tracking_ang_vel": r_tracking_ang_vel,
            "tracking_orientation": r_tracking_orientation,
            "lin_vel_z": r_lin_vel_z,
            "ang_vel_xy": r_ang_vel_xy,
            "orientation": r_orientation,
            "torques": r_torques,
            "joint_acceleration": r_joint_acceleration,
            "mechanical_work": r_mechanical_work,
            "action_rate": r_action_rate,
            "feet_air_time": r_feet_air_time,
            "stand_still": r_stand_still,
            "stand_still_joint_velocity": r_stand_still_joint_velocity,
            "abduction_angle": r_abduction_angle,
            "termination": r_termination,
            "foot_slip": r_foot_slip,
            "knee_collision": r_knee_collision,
            "body_collision": r_body_collision,
        }
        scaled = {k: REWARD_WEIGHTS[k] * v for k, v in terms.items()}
        reward = float(np.clip(sum(scaled.values()) * self.dt, 0.0, 10000.0))

        info = {f"r_{k}": float(scaled[k]) for k in scaled}
        return reward, info

    # ----------------- helpers -----------------

    def _gravity_in_body(self) -> np.ndarray:
        R = self.data.xmat[self._base_id].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _get_obs(self) -> np.ndarray:
        # IMU 数据：local angular vel + gravity in body
        R = self.data.xmat[self._base_id].reshape(3, 3)
        local_ang_vel = R.T @ self.data.qvel[3:6]
        gravity = R.T @ np.array([0.0, 0.0, -1.0])

        ang_noise = self.np_random.uniform(-1, 1, size=3) * ANG_VEL_NOISE
        grav_noise = self.np_random.uniform(-1, 1, size=3) * GRAVITY_NOISE
        motor_noise = self.np_random.uniform(-1, 1, size=12) * MOTOR_ANGLE_NOISE
        last_act_noise = self.np_random.uniform(-1, 1, size=12) * LAST_ACTION_NOISE

        noised_gravity = gravity + grav_noise
        noised_gravity = noised_gravity / np.linalg.norm(noised_gravity)
        noised_ang_vel = local_ang_vel + ang_noise
        noised_imu = np.concatenate([noised_ang_vel, noised_gravity])

        lagged_imu, self._imu_buffer = _sample_lagged(
            self.np_random, self._imu_buffer, noised_imu, IMU_LATENCY_DIST,
        )

        joint_angles = self.data.qpos[self._joint_qpos_ids]
        obs_step = np.concatenate([
            lagged_imu,                             # 6
            self.cmd.astype(np.float64),            # 3
            self.desired_world_z,                   # 3
            (joint_angles - DEFAULT_POSE) + motor_noise,   # 12
            self.last_action.astype(np.float64) + last_act_noise,   # 12
        ]).astype(np.float32)
        obs_step = np.clip(obs_step, -100.0, 100.0)

        # 滚动 stack
        self._obs_history = np.roll(self._obs_history, OBS_DIM_PER_STEP)
        self._obs_history[:OBS_DIM_PER_STEP] = obs_step
        return self._obs_history.copy()

    def _sample_command(self) -> np.ndarray:
        if self.np_random.uniform() < ZERO_CMD_PROB:
            cmd = self.np_random.uniform(
                -STAND_STILL_THRESHOLD, STAND_STILL_THRESHOLD, size=3,
            )
        else:
            vx = self.np_random.uniform(*CMD_VX_RANGE)
            vy = self.np_random.uniform(*CMD_VY_RANGE)
            wz = self.np_random.uniform(*CMD_WZ_RANGE)
            cmd = np.array([vx, vy, wz])
        return cmd.astype(np.float32)


def _sample_lagged(
    rng: np.random.Generator, buffer_newest_first: np.ndarray, new_value: np.ndarray, distribution: np.ndarray,
):
    """Push `new_value` to front of buffer, sample one column weighted by `distribution`."""
    buf = np.roll(buffer_newest_first, shift=1, axis=1)
    buf[:, 0] = new_value
    idx = rng.choice(buf.shape[1], p=distribution)
    return buf[:, idx].copy(), buf


class FrameStackedPupperEnv(gym.Env):
    """Backward-compat wrapper.

    Bar D 版本通过这个 wrapper 把 49-dim flat obs stack 15 帧。本版本 `PupperEnv`
    内部已经做了 15 帧 stack（与 MJX/brax 版一致），所以这个 wrapper 退化为
    透明 passthrough，保留接口避免破坏 starter.py 等老调用点。
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, n_stack: int = N_STACK, **inner_kwargs):
        super().__init__()
        if n_stack != N_STACK:
            raise ValueError(
                f"PupperEnv 内置 stack={N_STACK} 帧；FrameStackedPupperEnv "
                f"传 n_stack={n_stack} 没意义。请直接用 PupperEnv 或保持默认。"
            )
        self.inner = PupperEnv(**inner_kwargs)
        self.action_space = self.inner.action_space
        self.observation_space = self.inner.observation_space

    def reset(self, *, seed=None, options=None):
        return self.inner.reset(seed=seed, options=options)

    def step(self, action):
        return self.inner.step(action)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.__dict__["inner"], name)
