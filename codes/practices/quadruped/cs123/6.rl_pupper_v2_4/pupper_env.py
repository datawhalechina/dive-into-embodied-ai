"""精简版 Pupper RL 环境。

- 基础观测：48 维，包含角速度、重力方向、命令、关节状态、上一步动作和机身线速度。
- 步态观测：可选追加 gait one-hot 和相位 sin/cos，共 53 维。
- 动作：12 维关节位置残差，先过一阶低通滤波（截止约 5.5 Hz）再由 PD 位置
  伺服器执行；训练与评估同样经过滤波。
- 奖励：速度跟踪、姿态、能耗、平滑性、足端腾空时间（目标 0.25 秒，抑制
  高频小碎步）、摆动相足端高度整形（foot_clearance，目标 3.5cm，产生可见
  的抬腿动作）、钉腿超时惩罚（feet_stance_time，防三腿跛行）、防抖动与
  防打滑惩罚、站立静止约束、拒绝执行命令惩罚（dont_wait）、可选步态接触
  和终止惩罚。
- 命令：每回合随机采样，回合内默认每 250 步重采样一次，训练命令切换的鲁棒性。
- 扰动：默认开启随机初始速度和偶发 kick，帮助逃出"站桩不动"局部最优并
  提高恢复能力；评估时应传 perturb_enabled=False。
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
FOOT_BODIES = [
    "leg_front_r_3", "leg_front_l_3",
    "leg_back_r_3", "leg_back_l_3",
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
# 策略输出先过一阶低通（EMA）再进 PD：α=0.5 在 50 Hz 下截止约 5.5 Hz，
# 保留步态基频和摆动相（1–5 Hz），滤掉网络输出的高频抖动。这是手写步态
# "天然平滑"的等价先验——限制动作频带；训练时同样经过滤波，策略学会穿过
# 它工作。α=1.0 等价于关闭滤波（退化为 v2.1 行为）。
ACTION_FILTER_ALPHA = 0.5
MAX_STEPS = 1000
FOOT_RADIUS = 0.02

CMD_VX_RANGE = (-0.75, 0.75)
CMD_VY_RANGE = (-0.5, 0.5)
CMD_WZ_RANGE = (-2.0, 2.0)
# 提高零命令概率让策略充分学习站立；回合内定期重采样训练命令切换。
ZERO_CMD_PROB = 0.1
CMD_RESAMPLE_STEPS = 250

# sigma 收紧到 0.15：站着不动时残余跟踪奖励从 0.37 降到 0.19，
# 削弱"站桩领补贴"的局部最优；好的跟踪（误差 0.1 内）几乎不受影响。
TRACKING_SIGMA = 0.15
# 0.1 秒的腾空目标等于给高频小碎步发许可证（5 Hz 点地也能领满奖励）；
# 提到 0.25 秒奖励完整摆动相，步频落回 2–3 Hz，接近 trot 时钟的 2 Hz。
AIR_TIME_TARGET = 0.25
# 摆动相足端目标离地高度，对齐 5.gait-control 的 step_height（3.0–4.5cm）。
# 无此项时贴地几毫米掠过是能量最优解，看不到"抬腿"动作，真实地形会绊倒。
FOOT_CLEARANCE_TARGET = 0.035
# 单脚连续触地超过该时长（且命令非零）开始按步惩罚，防止某条腿钉死当锚、
# 其余三腿跛行；正常步态的单脚支撑相约 0.2 秒，不受影响。
STANCE_TIME_LIMIT = 0.4
CMD_DEAD_ZONE = 0.05
# dont_wait 判定：命令非零但四足全触地且机身几乎不动时罚分。
WAIT_LIN_SPEED = 0.1
WAIT_ANG_SPEED = 0.3
# penalty 项 term 返回正幅值，权重写负。项目集合与 exercises/lab_6_rl_pupper 对齐；
# 平滑类惩罚比完整版更温和、腾空奖励更高：20M 步预算下过重的学步期税负
# 会让策略收敛到"冻结不动"的局部最优（总奖励下限裁剪为 0，尝试迈步反而不如静止）。
REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.5,
    "tracking_ang_vel": 0.8,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "torques": -2e-4,
    "joint_acceleration": -2.5e-7,
    "action_rate": -0.01,
    "feet_air_time": 1.0,
    "feet_stance_time": -1.0,
    "foot_clearance": -100.0,
    "foot_slip": -0.05,
    "abduction_angle": -0.1,
    "stand_still": -0.5,
    "stand_still_joint_velocity": -0.1,
    "dont_wait": -1.0,
    "termination": -100.0,
}

# 默认姿态的自然站高约 0.154；终止线从 0.10 抬到 0.12 排除"蜷伏冻结"局部
# 最优，同时给步态中的机身起伏保留余量（抬到 0.13 会误杀探索期的正常踉跄）。
TERMINAL_BODY_Z = 0.12
TERMINAL_BODY_ANGLE = 0.52

INIT_HEIGHT = 0.28
JOINT_INIT_NOISE = 0.05
# 扰动帮助逃出"站桩不动"局部最优：开局带初速度迫使策略从第一步就学迈步，
# 偶发 kick 迫使站立策略学会迈步恢复。kick 参数与 exercises/lab_6_rl_pupper 一致。
INIT_VEL_NOISE = 0.3
INIT_YAW_RATE_NOISE = 0.5
KICK_PROBABILITY = 0.02
KICK_VEL = 0.2
STAND_STILL_THRESHOLD = 0.1
DESIRED_ABDUCTION = np.zeros(4, dtype=np.float64)
SLIP_CONTACT_HEIGHT = 3e-2
BASE_OBS_DIM = 48
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
        cmd_resample_steps: int = CMD_RESAMPLE_STEPS,
        perturb_enabled: bool = True,
        action_filter_alpha: float = ACTION_FILTER_ALPHA,
        reward_overrides: dict[str, float] | None = None,
        gait_enabled: bool = False,
        gait_types: tuple[str, ...] | list[str] = GAIT_NAMES,
        gait_contact_weight: float = 0.5,
        gait_switch_steps: int = 500,
    ):
        super().__init__()
        overrides = dict(reward_overrides or {})
        unknown_terms = sorted(set(overrides) - set(REWARD_WEIGHTS))
        if unknown_terms:
            raise ValueError(f"未知奖励项：{', '.join(unknown_terms)}")
        self.reward_weights = {**REWARD_WEIGHTS, **overrides}
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT_PHYSICS
        self.dt = DT_CONTROL
        self.n_substeps = max(1, round(DT_CONTROL / DT_PHYSICS))
        self.max_steps = max_steps
        self.cmd_resample_steps = int(cmd_resample_steps)
        if self.cmd_resample_steps < 0:
            raise ValueError("cmd_resample_steps 不能小于 0")
        self.perturb_enabled = bool(perturb_enabled)
        self.action_filter_alpha = float(action_filter_alpha)
        if not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha 必须在 (0, 1] 内")
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
        self._foot_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ], dtype=int)

        # torque = KP * (ctrl - qpos) - KD * qvel
        self.model.actuator_gainprm[:, 0] = KP
        self.model.actuator_biasprm[:, 1] = -KP
        self.model.actuator_biasprm[:, 2] = -KD

        self.last_action = np.zeros(12, dtype=np.float32)
        self.filtered_action = np.zeros(12, dtype=np.float64)
        self.last_joint_vel = np.zeros(12, dtype=np.float64)
        self.feet_air_time = np.zeros(4, dtype=np.float64)
        self.feet_stance_time = np.zeros(4, dtype=np.float64)
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
        if self.perturb_enabled:
            self.data.qvel[0:2] = self.np_random.uniform(
                -INIT_VEL_NOISE, INIT_VEL_NOISE, size=2,
            )
            self.data.qvel[5] = self.np_random.uniform(
                -INIT_YAW_RATE_NOISE, INIT_YAW_RATE_NOISE,
            )
        self.data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.filtered_action[:] = 0.0
        self.last_joint_vel[:] = 0.0
        self.feet_air_time[:] = 0.0
        self.feet_stance_time[:] = 0.0
        self.last_contact[:] = False
        self.cmd = self._sample_command()
        self.step_count = 0
        if self.gait_enabled:
            self._sample_gait()
            self.gait_phase = float(self.np_random.uniform(0.0, 1.0))
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        if (
            self.perturb_enabled
            and self.np_random.uniform() < KICK_PROBABILITY
        ):
            self.data.qvel[0:2] += self.np_random.uniform(
                -KICK_VEL, KICK_VEL, size=2,
            )
        self.filtered_action = (
            self.action_filter_alpha * action.astype(np.float64)
            + (1.0 - self.action_filter_alpha) * self.filtered_action
        )
        motor_target = np.clip(
            DEFAULT_POSE + ACTION_SCALE * self.filtered_action,
            JOINT_LOWERS,
            JOINT_UPPERS,
        )
        self.data.ctrl[:] = motor_target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        foot_z = self.data.site_xpos[self._foot_site_ids][:, 2] - FOOT_RADIUS
        contact = foot_z < 1e-3
        # 打滑惩罚使用更宽松的接触判定，覆盖足端贴地拖动的情形。
        slip_contact = (foot_z < SLIP_CONTACT_HEIGHT) | self.last_contact
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.dt
        self.feet_stance_time += self.dt

        rotation = self.data.xmat[self._base_id].reshape(3, 3)
        up_in_body_z = float(
            (rotation.T @ np.array([0.0, 0.0, 1.0]))[2],
        )
        terminated = bool(
            self.data.qpos[2] < TERMINAL_BODY_Z
            or up_in_body_z < np.cos(TERMINAL_BODY_ANGLE)
        )

        reward, info = self._compute_reward(
            action, rotation, contact, slip_contact, first_contact,
            foot_z, terminated,
        )

        self.feet_air_time *= (~contact).astype(np.float64)
        self.feet_stance_time *= contact.astype(np.float64)
        self.last_contact = contact
        self.last_action = action.copy()
        self.last_joint_vel = self.data.qvel[self._qvel_ids].copy()
        self.step_count += 1
        if (
            self.cmd_resample_steps > 0
            and self.step_count % self.cmd_resample_steps == 0
            and self.step_count < self.max_steps
        ):
            self.cmd = self._sample_command()
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
        slip_contact,
        first_contact,
        foot_z,
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
        joint_angles = self.data.qpos[self._qpos_ids]
        joint_vel = self.data.qvel[self._qvel_ids]

        if cmd_magnitude > CMD_DEAD_ZONE:
            feet_air_time = float(np.sum(
                (self.feet_air_time - AIR_TIME_TARGET) * first_contact,
            ))
            # 与 feet_air_time 互补：抬腿给奖励只能约束勤快的腿，
            # 钉死不抬的腿必须按超时持续罚，否则策略会学成三腿跛行。
            feet_stance_time = float(np.sum(
                (self.feet_stance_time > STANCE_TIME_LIMIT) & contact,
            ))
        else:
            feet_air_time = 0.0
            feet_stance_time = 0.0

        # 接触中的足端水平速度视为打滑。cvel 布局为 [角速度, 线速度]。
        foot_slip = 0.0
        for index, body_id in enumerate(self._foot_body_ids):
            if slip_contact[index]:
                foot_vel = self.data.cvel[body_id, 3:5]
                foot_slip += float(foot_vel[0] ** 2 + foot_vel[1] ** 2)

        # 摆动相足端离目标高度的偏差，按水平速度加权：贴地掠过或蹭地都被罚，
        # 摆到目标高度（约手写步态的 step_height）零罚。
        foot_clearance = 0.0
        if cmd_magnitude > CMD_DEAD_ZONE:
            for index, body_id in enumerate(self._foot_body_ids):
                if not contact[index]:
                    foot_vel = self.data.cvel[body_id, 3:5]
                    speed = float(np.hypot(foot_vel[0], foot_vel[1]))
                    height_error = float(foot_z[index]) - FOOT_CLEARANCE_TARGET
                    foot_clearance += height_error ** 2 * speed

        if cmd_magnitude < STAND_STILL_THRESHOLD:
            stand_still = float(np.sum(np.abs(joint_angles - DEFAULT_POSE)))
            stand_still_joint_velocity = float(np.sum(np.abs(joint_vel)))
        else:
            stand_still = 0.0
            stand_still_joint_velocity = 0.0

        # 命令要求移动而策略站着不动：罚掉"站桩领残余跟踪分"的局部最优。
        waiting = (
            cmd_magnitude > CMD_DEAD_ZONE
            and bool(np.all(contact))
            and float(np.linalg.norm(local_lin[0:2])) < WAIT_LIN_SPEED
            and abs(float(local_ang[2])) < WAIT_ANG_SPEED
        )
        dont_wait = 1.0 if waiting else 0.0

        terms = {
            "tracking_lin_vel": float(np.exp(-lin_error / TRACKING_SIGMA)),
            "tracking_ang_vel": float(np.exp(-ang_error / TRACKING_SIGMA)),
            "lin_vel_z": float(local_lin[2] ** 2),
            "ang_vel_xy": float(local_ang[0] ** 2 + local_ang[1] ** 2),
            "orientation": float(up[0] ** 2 + up[1] ** 2),
            "torques": float(np.sum(torques ** 2)),
            "joint_acceleration": float(np.sum(
                ((joint_vel - self.last_joint_vel) / self.dt) ** 2,
            )),
            "action_rate": float(np.sum(
                (action.astype(np.float64) - self.last_action) ** 2,
            )),
            "feet_air_time": feet_air_time,
            "feet_stance_time": feet_stance_time,
            "foot_clearance": foot_clearance,
            "foot_slip": foot_slip,
            "abduction_angle": float(np.sum(
                (joint_angles[1::3] - DESIRED_ABDUCTION) ** 2,
            )),
            "stand_still": stand_still,
            "stand_still_joint_velocity": stand_still_joint_velocity,
            "dont_wait": dont_wait,
            "termination": 1.0 if terminated else 0.0,
        }
        scaled = {
            key: self.reward_weights[key] * value
            for key, value in terms.items()
        }
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
        lin_vel = rotation.T @ self.data.qvel[0:3]
        ang_vel = rotation.T @ self.data.qvel[3:6]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm > 1e-9:
            gravity /= gravity_norm

        joint_angles = self.data.qpos[self._qpos_ids]
        joint_vel = self.data.qvel[self._qvel_ids]
        # 机身线速度追加在末尾，保证旧 45 维 checkpoint 迁移时前缀语义不变。
        obs = np.concatenate([
            ang_vel,
            gravity,
            self.cmd.astype(np.float64),
            joint_angles - DEFAULT_POSE,
            joint_vel,
            self.last_action.astype(np.float64),
            lin_vel,
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
