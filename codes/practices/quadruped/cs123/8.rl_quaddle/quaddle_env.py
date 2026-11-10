"""Quaddle RL 环境 v2：CPG 参数化动作空间。

v1 教训（outputs/train_v2.log）：逐时刻 4 维关节动作下 PPO 2M 步仍站桩
（|vx|≈6mm/s）——策略需要自己发明 2-4Hz 相干振荡，而观测中没有相位时钟，
探索效率极低。诊断实验（任意相位差 CPG 扫描）证明物理可行解遍地都是
（20~190mm/s），缺的只是"周期性"这个结构先验。

v2 把动作空间升级为 5 个慢变步态参数（tanh ±1）：

    a[0] amp   : 摆幅 0~14°
    a[1] freq  : 0.5~4.0 Hz
    a[2] form  : 相位形，-1=wave(0/90/180/270) ↔ +1=anti-wave ↔ 0=trot(0/180/0/180)
    a[3] sign  : 行进方向（±）
    a[4] turn  : 左右侧摆幅差（转向）
    a[5] skew  : 波形二次谐波系数（步形前后偏斜 = 行进非对称性之源）

env 内部以 50Hz CPG 生成舵机目标，参数经 τ=0.15s 低通（慢变）。
观测 26 维（v1 的 24 维，last_action 从 4 维扩到 5 维）。
奖励沿用 v1：速度跟踪 + 存活 + dont_wait + 正则（action 罚改为参数变化率）。

物理 1 kHz，控制 50 Hz，回合 5 s。
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(LAB_DIR / ".." / "7.quaddle-mjcf" / "quaddle.xml")

LEGS = ("lf", "rf", "rb", "lb")

DT_PHYSICS = 0.001
DT_CONTROL = 0.02
N_SUBSTEPS = round(DT_CONTROL / DT_PHYSICS)  # 20

AMP_MAX_DEG = 14.0
FREQ_MIN_HZ, FREQ_MAX_HZ = 0.5, 4.0
PARAM_FILTER_TAU = 0.15      # 步态参数低通（慢变）
CTRL_LIMIT = 2.44            # actuator ctrlrange（±140°）
MAX_STEPS = 250              # 5 s / 回合

CMD_VX_RANGE = (-0.12, 0.12)
CMD_VY_RANGE = (-0.08, 0.08)
CMD_WZ_RANGE = (-1.2, 1.2)
ZERO_CMD_PROB = 0.1
CMD_RESAMPLE_STEPS = 125     # 2.5 s 重采样

TRACKING_LIN_SIGMA = 0.08
TRACKING_ANG_SIGMA = 0.5

REWARD_WEIGHTS = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 1.0,
    "alive": 0.5,
    "lin_vel_z": -1.0,
    "orientation": -2.5,
    "action_rate": -0.05,      # 对象变为步态参数变化量（量级更小，权重加大）
    "dont_wait": -1.5,
    "termination": -50.0,
}

# dont_wait 判定（v2.5 经验移植）：命令非零但原地钉死
CMD_DEAD_ZONE = 0.05
WAIT_LIN_SPEED = 0.02
WAIT_ANG_SPEED = 0.2

TERMINAL_BODY_Z = 0.030      # 沉降站高 ~0.0495
TERMINAL_BODY_ANGLE = 0.52   # ≈30°

INIT_HEIGHT = 0.06
INIT_VEL_NOISE = 0.15
INIT_YAW_RATE_NOISE = 0.4
KICK_PROBABILITY = 0.02
KICK_VEL = 0.15

OBS_DIM = 26                 # 21(状态+命令) + 关节6 + 步态参数 6
ACT_DIM = 6               # amp/freq/form/sign/turn/skew

# 相位形基组（弧度）：trot / wave / anti-wave
_PHASE_TROT = np.array([0.0, np.pi, 0.0, np.pi])
_PHASE_WAVE = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
_PHASE_ANTI = np.array([0.0, -0.5 * np.pi, np.pi, -1.5 * np.pi])


def _phase_set(form: float) -> np.ndarray:
    """form∈[-1,1]：0=trot，-1=wave，+1=anti-wave，中间线性插值。"""
    if form < 0:
        t = -form
        return (1 - t) * _PHASE_TROT + t * _PHASE_WAVE
    t = form
    return (1 - t) * _PHASE_TROT + t * _PHASE_ANTI


class QuaddleEnv(gym.Env):
    """可供 Stable-Baselines3 使用的 Quaddle Gymnasium 环境（CPG 动作空间）。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml: str = MODEL_PATH,
        max_steps: int = MAX_STEPS,
        cmd_resample_steps: int = CMD_RESAMPLE_STEPS,
        perturb_enabled: bool = True,
        reward_overrides: dict[str, float] | None = None,
    ):
        super().__init__()
        overrides = dict(reward_overrides or {})
        unknown = sorted(set(overrides) - set(REWARD_WEIGHTS))
        if unknown:
            raise ValueError(f"未知奖励项：{', '.join(unknown)}")
        self.reward_weights = {**REWARD_WEIGHTS, **overrides}

        self.model = mujoco.MjModel.from_xml_path(xml)
        self.model.opt.timestep = DT_PHYSICS
        self.data = mujoco.MjData(self.model)
        self.dt = DT_CONTROL
        self.n_substeps = N_SUBSTEPS
        self.max_steps = int(max_steps)
        self.cmd_resample_steps = int(cmd_resample_steps)
        self.perturb_enabled = bool(perturb_enabled)

        self._body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )

        self._cmd = np.zeros(3)
        self._params = np.zeros(ACT_DIM)       # 滤波后的步态参数
        self._last_params = np.zeros(ACT_DIM)
        self._phase = 0.0                      # CPG 相位时钟（秒）
        self._step_count = 0
        self._steps_since_resample = 0

    # -- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        qpos = self.data.qpos.copy()
        if self.perturb_enabled:
            qpos[2] = INIT_HEIGHT
            # 保留 home 的 yaw90 四元数（前进=世界+x），小幅扰动关节
            qpos[7:] += self.np_random.uniform(-0.01, 0.01, size=24)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.perturb_enabled:
            self.data.qvel[:3] += self.np_random.uniform(
                -INIT_VEL_NOISE, INIT_VEL_NOISE, size=3
            )
            self.data.qvel[5] += self.np_random.uniform(
                -INIT_YAW_RATE_NOISE, INIT_YAW_RATE_NOISE
            )
        self._cmd[:] = 0.0
        self._resample_command()
        self._params[:] = 0.0
        self._last_params[:] = 0.0
        self._phase = 0.0
        self._step_count = 0
        self._steps_since_resample = 0
        mujoco.mj_forward(self.model, self.data)
        # 物理沉降 1.2s：吸收关节噪声、落地弹跳（0.5s 时弹跳未衰减曾致悬空翻倒）
        self.data.ctrl[:] = 0.0
        for _ in range(int(1.2 / DT_PHYSICS)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._observe(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        self._last_params[:] = self._params
        alpha = 1.0 - np.exp(-self.dt / PARAM_FILTER_TAU)
        self._params += alpha * (action - self._params)

        amp01, freq01, form, sign, turn, skew = self._params
        amp = math.radians((amp01 + 1.0) / 2.0 * AMP_MAX_DEG)   # 0~14°
        freq = FREQ_MIN_HZ + (freq01 + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ)
        self._phase += 2.0 * np.pi * freq * self.dt
        s = self._phase * (1.0 if sign >= 0 else -1.0)
        phases = _phase_set(form)
        turn = np.clip(turn, -1.0, 1.0)
        skew = 0.45 * np.clip(skew, -1.0, 1.0)
        amps = amp * np.array([1.0 + 0.6 * turn, 1.0 - 0.6 * turn,
                               1.0 + 0.6 * turn, 1.0 - 0.6 * turn])
        wave = np.sin(s + phases) + skew * np.sin(2.0 * (s + phases))
        targets = np.clip(amps * wave, -CTRL_LIMIT, CTRL_LIMIT)
        self.data.ctrl[:] = targets

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self._steps_since_resample += 1
        if (
            self.cmd_resample_steps > 0
            and self._steps_since_resample >= self.cmd_resample_steps
        ):
            self._resample_command()
            self._steps_since_resample = 0
        if self.perturb_enabled and self.np_random.random() < KICK_PROBABILITY:
            self.data.qvel[:3] += self.np_random.uniform(
                -KICK_VEL, KICK_VEL, size=3
            )

        obs = self._observe()
        terminated, reason = self._check_termination()
        reward = self._compute_reward(terminated)
        truncated = self._step_count >= self.max_steps
        info = {"cmd": self._cmd.copy(), "termination": reason, "gait_params": self._params.copy()}
        return obs, float(reward), terminated, truncated, info

    # -- 内部 ------------------------------------------------------------

    def _resample_command(self) -> None:
        if self.np_random.random() < ZERO_CMD_PROB:
            self._cmd[:] = 0.0
            return
        self._cmd[:] = np.array([
            self.np_random.uniform(*CMD_VX_RANGE),
            self.np_random.uniform(*CMD_VY_RANGE),
            self.np_random.uniform(*CMD_WZ_RANGE),
        ])

    def _body_states(self) -> dict[str, np.ndarray]:
        r = self.data.body(self._body_id).xmat.reshape(3, 3).copy()
        r_t = r.T
        return {
            "gravity": r_t @ np.array([0.0, 0.0, -1.0]),
            "ang_vel": r_t @ self.data.qvel[3:6],
            "lin_vel": r_t @ self.data.qvel[:3],
            "rpy_mat": r,
        }

    def _observe(self) -> np.ndarray:
        s = self._body_states()
        joint_q = np.array([self.data.qpos[7 + i * 6] for i in range(4)])
        joint_v = np.array([self.data.qvel[6 + i * 6] for i in range(4)])
        return np.concatenate([
            s["gravity"],          # 3
            s["ang_vel"],          # 3
            s["lin_vel"],          # 3
            self._cmd,             # 3
            joint_q,               # 4
            joint_v,               # 4
            self._params,          # 5（当前步态参数）
        ])

    def _feet_on_ground(self) -> int:
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_col")
            for leg in LEGS
        }
        n = 0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g = (con.geom1, con.geom2)
            if floor_id in g and any(fid in g for fid in foot_ids):
                n += 1
        return n

    def _check_termination(self) -> tuple[bool, str]:
        s = self._body_states()
        mat = s["rpy_mat"]
        roll = abs(np.arctan2(mat[2, 1], mat[2, 2]))
        pitch = abs(np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2])))
        if roll > TERMINAL_BODY_ANGLE or pitch > TERMINAL_BODY_ANGLE:
            return True, "tip_over"
        z = float(self.data.body(self._body_id).xpos[2])
        if z < TERMINAL_BODY_Z:
            return True, "collapse"
        return False, ""

    def _compute_reward(self, terminated: bool) -> float:
        s = self._body_states()
        w = self.reward_weights
        if terminated:
            return float(w["termination"])
        vel = s["lin_vel"]
        lin_err = (self._cmd[0] - vel[0]) ** 2 + (self._cmd[1] - vel[1]) ** 2
        ang_err = (self._cmd[2] - s["ang_vel"][2]) ** 2
        mat = s["rpy_mat"]
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2]))
        cmd_lin = np.hypot(self._cmd[0], self._cmd[1])
        waiting = (
            cmd_lin > CMD_DEAD_ZONE
            and np.hypot(vel[0], vel[1]) < WAIT_LIN_SPEED
            and abs(s["ang_vel"][2]) < WAIT_ANG_SPEED
        )
        terms = {
            "tracking_lin_vel": np.exp(-lin_err / (2 * TRACKING_LIN_SIGMA**2)),
            "tracking_ang_vel": np.exp(-ang_err / (2 * TRACKING_ANG_SIGMA**2)),
            "alive": 1.0,
            "lin_vel_z": float(vel[2] ** 2),
            "orientation": float(roll**2 + pitch**2),
            "action_rate": float(np.sum((self._params - self._last_params) ** 2)),
            "dont_wait": 1.0 if waiting else 0.0,
        }
        return float(sum(w[k] * v for k, v in terms.items()))

