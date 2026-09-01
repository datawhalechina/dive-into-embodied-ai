"""精简版 Pupper RL 环境。

- 基础观测：45 维，包含角速度、重力方向、命令、关节状态和上一步动作。
- 步态观测：可选追加 gait one-hot 和相位 sin/cos，共 50 维。
- 动作：12 维关节位置残差，由 PD 位置伺服器执行。
- 奖励：速度跟踪、姿态、能耗、平滑性、足端腾空时间、可选步态接触和终止惩罚。
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
ZERO_CMD_PROB = 0.01

TRACKING_SIGMA = 0.25
AIR_TIME_TARGET = 0.1
CMD_DEAD_ZONE = 0.05
REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.5,
    "tracking_ang_vel": 0.8,
    "orientation": -5.0,
    "torques": -2e-4,
    "action_rate": -0.01,
    "feet_air_time": 0.2,
    "termination": -100.0,
}

TERMINAL_BODY_Z = 0.10
TERMINAL_BODY_ANGLE = 0.52

INIT_HEIGHT = 0.28
JOINT_INIT_NOISE = 0.05
STAND_STILL_THRESHOLD = 0.1
BASE_OBS_DIM = 45
GAIT_FEATURE_DIM = 5
GAIT_NAMES = ("walk", "trot", "pace")
GAIT_SPECS = {
    # 足端顺序为 FR、FL、RR、RL；offset 表示各足进入支撑相的周期位置。
    "walk": {
        "offsets": np.array([0.75, 0.25, 0.50, 0.00]),
        "duty_factor": 0.75,
        "cycle_time": 0.90,
    },
    "trot": {
        "offsets": np.array([0.50, 0.00, 0.00, 0.50]),
        "duty_factor": 0.50,
        "cycle_time": 0.50,
    },
    "pace": {
        "offsets": np.array([0.50, 0.00, 0.50, 0.00]),
        "duty_factor": 0.50,
        "cycle_time": 0.50,
    },
}


class PupperEnv(gym.Env):
    """可供 Stable-Baselines3 使用的 Pupper Gymnasium 环境。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml: str = MODEL_PATH,
        max_steps: int = MAX_STEPS,
        gait_enabled: bool = False,
        gait_types: tuple[str, ...] | list[str] = GAIT_NAMES,
        gait_contact_weight: float = 0.5,
        gait_switch_steps: int = 500,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT_PHYSICS
        self.dt = DT_CONTROL
        self.n_substeps = max(1, round(DT_CONTROL / DT_PHYSICS))
        self.max_steps = max_steps
        self.gait_enabled = bool(gait_enabled)
        self.gait_types = tuple(gait_types)
        unknown_gaits = sorted(set(self.gait_types) - set(GAIT_NAMES))
        if not self.gait_types:
            raise ValueError("gait_types 不能为空")
        if unknown_gaits:
            raise ValueError(f"未知步态：{', '.join(unknown_gaits)}")
        self.gait_contact_weight = float(gait_contact_weight)
        self.gait_switch_steps = int(gait_switch_steps)
        if self.gait_switch_steps < 0:
            raise ValueError("gait_switch_steps 不能小于 0")

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        obs_dim = BASE_OBS_DIM + (GAIT_FEATURE_DIM if self.gait_enabled else 0)
        self.observation_space = spaces.Box(-100.0, 100.0, (obs_dim,), np.float32)

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
        self.gait_name = self.gait_types[0]
        self.gait_phase = 0.0

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
        if self.gait_enabled:
            self._sample_gait()
            self.gait_phase = float(self.np_random.uniform(0.0, 1.0))
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
            action, rotation, contact, first_contact, terminated,
        )

        self.feet_air_time *= (~contact).astype(np.float64)
        self.last_contact = contact
        self.last_action = action.copy()
        self.step_count += 1
        if self.gait_enabled:
            cycle_time = float(GAIT_SPECS[self.gait_name]["cycle_time"])
            self.gait_phase = (self.gait_phase + self.dt / cycle_time) % 1.0
            if (
                self.gait_switch_steps > 0
                and self.step_count % self.gait_switch_steps == 0
                and self.step_count < self.max_steps
            ):
                self._sample_gait(exclude_current=True)
                self.gait_phase = 0.0
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(
        self,
        action,
        rotation,
        contact,
        first_contact,
        terminated,
    ):
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
            "tracking_lin_vel": float(np.exp(-lin_error / TRACKING_SIGMA)),
            "tracking_ang_vel": float(np.exp(-ang_error / TRACKING_SIGMA)),
            "orientation": float(up[0] ** 2 + up[1] ** 2),
            "torques": float(np.sum(torques ** 2)),
            "action_rate": float(np.sum(
                (action.astype(np.float64) - self.last_action) ** 2,
            )),
            "feet_air_time": feet_air_time,
            "termination": 1.0 if terminated else 0.0,
        }
        scaled = {key: REWARD_WEIGHTS[key] * value for key, value in terms.items()}
        if self.gait_enabled:
            expected_contact = self._expected_contacts()
            contact_match = float(np.mean(contact == expected_contact))
            scaled["gait_contact"] = self.gait_contact_weight * contact_match
        reward = float(np.clip(sum(scaled.values()) * self.dt, 0.0, 10000.0))
        info = {f"r_{key}": float(value) for key, value in scaled.items()}
        if self.gait_enabled:
            info.update({
                "gait_name": self.gait_name,
                "gait_phase": float(self.gait_phase),
                "gait_contact_match": contact_match,
                "expected_contacts": expected_contact.astype(np.int8).tolist(),
                "actual_contacts": contact.astype(np.int8).tolist(),
            })
        return reward, info

    def _get_obs(self):
        rotation = self.data.xmat[self._base_id].reshape(3, 3)
        ang_vel = rotation.T @ self.data.qvel[3:6]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm > 1e-9:
            gravity /= gravity_norm

        joint_angles = self.data.qpos[self._qpos_ids]
        joint_vel = self.data.qvel[self._qvel_ids]
        obs = np.concatenate([
            ang_vel,
            gravity,
            self.cmd.astype(np.float64),
            joint_angles - DEFAULT_POSE,
            joint_vel,
            self.last_action.astype(np.float64),
        ])
        if self.gait_enabled:
            gait_one_hot = np.zeros(len(GAIT_NAMES), dtype=np.float64)
            gait_one_hot[GAIT_NAMES.index(self.gait_name)] = 1.0
            phase_angle = 2.0 * np.pi * self.gait_phase
            obs = np.concatenate([
                obs,
                gait_one_hot,
                [np.sin(phase_angle), np.cos(phase_angle)],
            ])
        obs = obs.astype(np.float32)
        return np.clip(obs, -100.0, 100.0)

    def set_gait(self, gait_name: str, *, reset_phase: bool = True) -> None:
        """固定当前步态，供评估和交互演示使用。"""
        if gait_name not in GAIT_NAMES:
            raise ValueError(f"未知步态：{gait_name}")
        if gait_name not in self.gait_types:
            raise ValueError(f"步态未在 gait_types 中启用：{gait_name}")
        self.gait_name = gait_name
        if reset_phase:
            self.gait_phase = 0.0

    def _sample_gait(self, *, exclude_current: bool = False) -> None:
        candidates = self.gait_types
        if exclude_current and len(candidates) > 1:
            candidates = tuple(name for name in candidates if name != self.gait_name)
        self.gait_name = str(self.np_random.choice(candidates))

    def _expected_contacts(self) -> np.ndarray:
        spec = GAIT_SPECS[self.gait_name]
        local_phase = (
            self.gait_phase - np.asarray(spec["offsets"], dtype=np.float64)
        ) % 1.0
        return local_phase < float(spec["duty_factor"])

    def _sample_command(self):
        if self.np_random.uniform() < ZERO_CMD_PROB:
            return self.np_random.uniform(
                -STAND_STILL_THRESHOLD,
                STAND_STILL_THRESHOLD,
                size=3,
            ).astype(np.float32)

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
