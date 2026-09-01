"""精简版 Pupper RL 环境。

- 观测：48 维，包含机身线/角速度、重力方向、命令、关节状态和上一步动作。
- 动作：12 维关节位置残差，由 PD 位置伺服器执行。
- 奖励：速度跟踪、姿态、能耗、平滑性、足端腾空时间和终止惩罚。
- 物理频率 250 Hz，控制频率 50 Hz。
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(LAB_DIR.parent / "assets" / "mjcfs" / "pupper_v3.xml")

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
    "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
]
FOOT_SITES = [
    "leg_front_r_3_foot_site", "leg_front_l_3_foot_site",
    "leg_back_r_3_foot_site", "leg_back_l_3_foot_site",
]

DEFAULT_POSE = np.array(
    [0.26, 0.0, -0.52, -0.26, 0.0, 0.52,
     0.26, 0.0, -0.52, -0.26, 0.0, 0.52],
    dtype=np.float64,
)
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

DT_PHYSICS = 0.004
DT_CONTROL = 0.02
KP, KD = 5.0, 0.25
ACTION_SCALE = 0.5
MAX_STEPS = 1000
FOOT_RADIUS = 0.02

CMD_VX_RANGE = (-0.75, 0.75)
CMD_VY_RANGE = (-0.5, 0.5)
CMD_WZ_RANGE = (-2.0, 2.0)
ZERO_CMD_PROB = 0.10

TRACKING_LIN_SIGMA = 0.10
TRACKING_ANG_SIGMA = 0.25
AIR_TIME_TARGET = 0.1
CMD_DEAD_ZONE = 0.05
REWARD_WEIGHTS = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 0.8,
    "lin_vel_error": -0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "torques": -2e-4,
    "action_rate": -0.01,
    "feet_air_time": 0.2,
    "stand_still": -0.5,
    "stand_still_joint_velocity": -0.05,
    "termination": -100.0,
}

TERMINAL_BODY_Z = 0.10
TERMINAL_BODY_ANGLE = 0.52

INIT_HEIGHT = 0.28
JOINT_INIT_NOISE = 0.05
STAND_STILL_THRESHOLD = 0.1


class PupperEnv(gym.Env):
    """可供 Stable-Baselines3 使用的 Pupper Gymnasium 环境。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml: str = MODEL_PATH, max_steps: int = MAX_STEPS):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT_PHYSICS
        self.dt = DT_CONTROL
        self.n_substeps = max(1, round(DT_CONTROL / DT_PHYSICS))
        self.max_steps = max_steps

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-100.0, 100.0, (48,), np.float32)

        self._base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link",
        )
        self._qpos_ids = np.array([
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name,
            )])
            for name in JOINT_NAMES
        ], dtype=int)
        self._qvel_ids = np.array([
            int(self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name,
            )])
            for name in JOINT_NAMES
        ], dtype=int)
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITES
        ], dtype=int)

        # torque = KP * (ctrl - qpos) - KD * qvel
        self.model.actuator_gainprm[:, 0] = KP
        self.model.actuator_biasprm[:, 1] = -KP
        self.model.actuator_biasprm[:, 2] = -KD

        self.last_action = np.zeros(12, dtype=np.float32)
        self.feet_air_time = np.zeros(4, dtype=np.float64)
        self.last_contact = np.zeros(4, dtype=bool)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0:3] = [0.0, 0.0, INIT_HEIGHT]
        self.data.qpos[3:7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2),
        ]
        self.data.qpos[self._qpos_ids] = (
            DEFAULT_POSE
            + self.np_random.uniform(-JOINT_INIT_NOISE, JOINT_INIT_NOISE, size=12)
        )
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.feet_air_time[:] = 0.0
        self.last_contact[:] = False
        self.cmd = self._sample_command()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        motor_target = np.clip(
            DEFAULT_POSE + ACTION_SCALE * action,
            JOINT_LOWERS,
            JOINT_UPPERS,
        )
        self.data.ctrl[:] = motor_target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        foot_z = self.data.site_xpos[self._foot_site_ids][:, 2] - FOOT_RADIUS
        contact = foot_z < 1e-3
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt

        rotation = self.data.xmat[self._base_id].reshape(3, 3)
        up_in_body_z = float(
            (rotation.T @ np.array([0.0, 0.0, 1.0]))[2],
        )
        terminated = bool(
            self.data.qpos[2] < TERMINAL_BODY_Z
            or up_in_body_z < np.cos(TERMINAL_BODY_ANGLE)
        )

        reward, info = self._compute_reward(
            action, rotation, first_contact, terminated,
        )

        self.feet_air_time *= (~contact).astype(np.float64)
        self.last_contact = contact
        self.last_action = action.copy()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, action, rotation, first_contact, terminated):
        local_lin = rotation.T @ self.data.qvel[0:3]
        local_ang = rotation.T @ self.data.qvel[3:6]
        vx, vy, wz = map(float, self.cmd)
        cmd_magnitude = float(np.linalg.norm(self.cmd))

        lin_error = (local_lin[0] - vx) ** 2 + (local_lin[1] - vy) ** 2
        ang_error = (local_ang[2] - wz) ** 2
        up = rotation @ np.array([0.0, 0.0, 1.0])
        torques = self.data.qfrc_actuator[self._qvel_ids]

        if cmd_magnitude > CMD_DEAD_ZONE:
            feet_air_time = float(np.sum(
                (self.feet_air_time - AIR_TIME_TARGET) * first_contact,
            ))
        else:
            feet_air_time = 0.0

        terms = {
            "tracking_lin_vel": float(
                np.exp(-lin_error / TRACKING_LIN_SIGMA),
            ),
            "tracking_ang_vel": float(
                np.exp(-ang_error / TRACKING_ANG_SIGMA),
            ),
            "lin_vel_error": float(lin_error),
            "lin_vel_z": float(local_lin[2] ** 2),
            "ang_vel_xy": float(local_ang[0] ** 2 + local_ang[1] ** 2),
            "orientation": float(up[0] ** 2 + up[1] ** 2),
            "torques": float(np.sum(torques ** 2)),
            "action_rate": float(np.sum(
                (action.astype(np.float64) - self.last_action) ** 2,
            )),
            "feet_air_time": feet_air_time,
            "stand_still": (
                float(np.sum(np.abs(
                    self.data.qpos[self._qpos_ids] - DEFAULT_POSE,
                )))
                if cmd_magnitude < STAND_STILL_THRESHOLD else 0.0
            ),
            "stand_still_joint_velocity": (
                float(np.sum(np.abs(self.data.qvel[self._qvel_ids])))
                if cmd_magnitude < STAND_STILL_THRESHOLD else 0.0
            ),
            "termination": 1.0 if terminated else 0.0,
        }
        scaled = {key: REWARD_WEIGHTS[key] * value for key, value in terms.items()}
        reward = float(np.clip(sum(scaled.values()) * self.dt, 0.0, 10000.0))
        info = {f"r_{key}": float(value) for key, value in scaled.items()}
        return reward, info

    def _get_obs(self):
        rotation = self.data.xmat[self._base_id].reshape(3, 3)
        lin_vel = rotation.T @ self.data.qvel[0:3]
        ang_vel = rotation.T @ self.data.qvel[3:6]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm > 1e-9:
            gravity /= gravity_norm

        joint_angles = self.data.qpos[self._qpos_ids]
        joint_vel = self.data.qvel[self._qvel_ids]
        obs = np.concatenate([
            lin_vel,
            ang_vel,
            gravity,
            self.cmd.astype(np.float64),
            joint_angles - DEFAULT_POSE,
            joint_vel,
            self.last_action.astype(np.float64),
        ]).astype(np.float32)
        return np.clip(obs, -100.0, 100.0)

    def _sample_command(self):
        if self.np_random.uniform() < ZERO_CMD_PROB:
            return np.zeros(3, dtype=np.float32)

        vx = self.np_random.uniform(*CMD_VX_RANGE)
        vy = self.np_random.uniform(*CMD_VY_RANGE)
        wz = self.np_random.uniform(*CMD_WZ_RANGE)
        return np.array([vx, vy, wz], dtype=np.float32)


if __name__ == "__main__":
    env = PupperEnv()
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape, "action space:", env.action_space)
    total_reward = 0.0
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(
            np.zeros(12, dtype=np.float32),
        )
        total_reward += reward
        if terminated or truncated:
            break
    print(
        f"100 steps zero-action: total reward = {total_reward:.3f}, "
        f"terminated = {terminated}",
    )
