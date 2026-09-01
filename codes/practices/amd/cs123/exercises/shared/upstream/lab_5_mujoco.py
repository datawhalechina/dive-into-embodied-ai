"""
CS123 Lab 5 — MuJoCo 版（Layer 1：console + plots）
"How to Train Your Dog"：本地 **不训练**，只负责把训练产物（`policy.json`）
拿过来做推理 / 拆解 / 分析。训练仍推荐走 Colab 或 `lab_5_train.py`（可选）。

运行方式：
    uv run python lab_5_mujoco.py

八个练习：
    Exercise 1 — Reward 函数库：8 大 reward 项的 NumPy 复刻 + "reward vs 变量" 图
    Exercise 2 — Dict → scalar：复现 pupperv3-mjx environment.py:484-490 的加权求和
    Exercise 3 — Observation layout：720 维 = 20 帧 × 36 维的精确解剖
    Exercise 4 — Policy 网络：parse `test_policy.json` / `parkour_policy.json`
    Exercise 5 — Pure-NumPy 推理：前向通过 + 与 viewer 的一致性校验
    Exercise 6 — Reward vs. gait：用 lab4 的 trot rollout 算 reward 分解
    Exercise 7 — Domain randomization：mass / friction / kp / push 的采样分布
    Exercise 8 — Sim2real gap：观测延迟 / kp mismatch / 力矩饱和的数字化

单位约定（和上游 pupperv3-mjx 保持一致，SI）：
    关节角 q                   [rad]
    关节角速度 qd              [rad/s]
    关节力矩 τ                 [N·m]   (MJCF forcerange = ±3)
    线速度 v, 角速度 ω         [m/s], [rad/s]
    时间 t                     [s]
    reward 系数 scale_i        量纲 = 1 / [term_i 的单位]，让 scale·term 的和无量纲
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# 中文 + 负号，和 lab3/lab4 一致
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
    "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

_DIR = pathlib.Path(__file__).resolve().parent
_SHARED_DIR = _DIR.parent
MODEL_FIXED = _SHARED_DIR / "models" / "pupper_v3_fixed.xml"
MODEL_FLOATING = _SHARED_DIR / "models" / "pupper_v3_floating.xml"
POLICY_DIR = _SHARED_DIR / "rl" / "policies"
PLOT_DIR = _DIR / "tmp" / "plots"

# -----------------------------------------------------------------------------
# 常量 —— 和 pupperv3-mjx 的 environment.py 默认值对齐
# -----------------------------------------------------------------------------

# 12 关节顺序（和 MJCF / neural_controller config.yaml / policy.json 都一致）。
JOINT_ORDER = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1",  "leg_back_r_2",  "leg_back_r_3",
    "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
]
LEG_NAMES = ["fr", "fl", "br", "bl"]

# Pupper "home" 姿态（两脚稍微折起来站着）—— 来自 environment.py:105 和 policy.json。
DEFAULT_POSE = np.array(
    [ 0.26, 0.0, -0.52, -0.26, 0.0,  0.52,
      0.26, 0.0, -0.52, -0.26, 0.0,  0.52], dtype=np.float64,
)

# 控制相关（neural_controller config.yaml 里的 `init_kps` / `init_kds`，实际训练时
# 会被 domain randomization 和 action_scale 盖掉；这里只是 baseline）。
NOMINAL_KP = 5.0          # [N·m/rad]  environment.py 默认 position_control_kp
NOMINAL_KD = 0.25         # [N·m·s/rad]  environment.py 默认 dof_damping
TORQUE_LIMIT = 3.0        # [N·m]      MJCF `forcerange="-3 3"`
CONTROL_DT = 0.02         # [s]        neural_controller 50 Hz = repeat_action(10) × 500 Hz
DEFAULT_TRACKING_SIGMA = 0.25   # 跟踪奖励 exp(-e²/σ) 的默认 σ（Lab 5 notebook 里典型值）

# Observation 布局（每帧 36 维；history=20 → 720 维 = policy.in_shape[1]）。
OBS_FIELDS: List[Tuple[str, int, str]] = [
    ("ang_vel",            3, "body 局部角速度 [rad/s]"),
    ("gravity",            3, "重力方向在 body 系投影（单位向量，带噪）"),
    ("command",            3, "速度指令 [vx, vy, wz]"),
    ("desired_world_z",    3, "期望 world-z 在 body 系（姿态指令）"),
    ("joint_pos_rel",     12, "关节位置 q − q_default [rad]"),
    ("last_action",       12, "上一步 policy 输出 a_{t-1} [−1,1]"),
]
OBS_PER_FRAME = sum(n for _, n, _ in OBS_FIELDS)   # 36
OBS_OFFSETS = {}
_off = 0
for name, n, _ in OBS_FIELDS:
    OBS_OFFSETS[name] = (_off, _off + n)
    _off += n
OBSERVATION_HISTORY = 20
OBS_TOTAL = OBS_PER_FRAME * OBSERVATION_HISTORY     # 720

# Domain randomization 范围（上游 domain_randomization.py 默认）。
DR_RANGES: Dict[str, Tuple[float, float]] = {
    "friction":           (0.6, 1.4),
    "kp_multiplier":      (0.75, 1.25),
    "kd_multiplier":      (0.5, 2.0),
    "body_com_x_shift":   (-0.03, 0.03),    # [m]
    "body_com_y_shift":   (-0.01, 0.01),
    "body_com_z_shift":   (-0.02, 0.02),
    "body_inertia_scale": (0.7, 1.3),
    "body_mass_scale":    (0.7, 1.3),
    "kick_velocity":      (0.0, 0.2),        # [m/s] 注入基座的脉冲
}


# =============================================================================
# Section A — Reward 函数库（NumPy 复刻 pupperv3_mjx/rewards.py）
# =============================================================================
# 每个函数都是 pure-function：输入 state/动作的原始张量，输出标量。
# 内部的 clip(-1000, 1000) 和上游对齐，避免极端值。

def _clip_reward(x: float) -> float:
    return float(np.clip(x, -1000.0, 1000.0))


def reward_tracking_lin_vel(
    command: np.ndarray, local_lin_vel: np.ndarray, sigma: float = DEFAULT_TRACKING_SIGMA,
) -> float:
    """指数跟踪：exp(-‖v_xy − cmd_xy‖² / σ)；越接近指令 reward 越高。"""
    e = float(np.sum((command[:2] - local_lin_vel[:2]) ** 2))
    return _clip_reward(np.exp(-e / (sigma + 1e-6)))


def reward_tracking_ang_vel(
    command: np.ndarray, local_ang_vel: np.ndarray, sigma: float = DEFAULT_TRACKING_SIGMA,
) -> float:
    """偏航率指数跟踪：exp(-(wz − cmd_wz)² / σ)。"""
    e = float((command[2] - local_ang_vel[2]) ** 2)
    return _clip_reward(np.exp(-e / (sigma + 1e-6)))


def reward_tracking_orientation(
    desired_world_z_in_body: np.ndarray, world_z_in_body: np.ndarray,
    sigma: float = DEFAULT_TRACKING_SIGMA,
) -> float:
    e = float(np.sum((world_z_in_body - desired_world_z_in_body) ** 2))
    return _clip_reward(np.exp(-e / (sigma + 1e-6)))


def reward_lin_vel_z(world_lin_vel: np.ndarray) -> float:
    """惩罚竖直方向的速度（不希望上下颠）。L2。"""
    return _clip_reward(world_lin_vel[2] ** 2)


def reward_ang_vel_xy(world_ang_vel: np.ndarray) -> float:
    """惩罚 roll / pitch 角速度（摇头晃脑）。L2 sum。"""
    return _clip_reward(float(np.sum(world_ang_vel[:2] ** 2)))


def reward_orientation(world_z_in_body: np.ndarray) -> float:
    """惩罚非水平姿态：‖(world_z_in_body)_xy‖²。完全直立时 world_z_in_body = [0,0,1]，分量 xy = 0。"""
    return _clip_reward(float(np.sum(world_z_in_body[:2] ** 2)))


def reward_torques(torques: np.ndarray) -> float:
    """L2 力矩惩罚（LeggedGym 风格）。"""
    return _clip_reward(float(np.sum(torques ** 2)))


def reward_joint_acceleration(qd: np.ndarray, last_qd: np.ndarray, dt: float = CONTROL_DT) -> float:
    return _clip_reward(float(np.sum(((qd - last_qd) / (dt + 1e-6)) ** 2)))


def reward_mechanical_work(torques: np.ndarray, qd: np.ndarray) -> float:
    """|τ · q̇| 的 L1。鼓励用最少的机械功做事（能量效率）。"""
    return _clip_reward(float(np.sum(np.abs(torques * qd))))


def reward_action_rate(action: np.ndarray, last_action: np.ndarray) -> float:
    """鼓励 action 变化平滑。‖Δa‖²。"""
    return _clip_reward(float(np.sum((action - last_action) ** 2)))


def reward_feet_air_time(
    air_time: np.ndarray, first_contact: np.ndarray, command: np.ndarray, minimum: float = 0.1,
) -> float:
    """刚触地那一步把 (air_time − 0.1) 计入 reward；鼓励长迈步。零命令时不给。"""
    r = float(np.sum((air_time - minimum) * first_contact))
    if float(np.linalg.norm(command[:3])) <= 0.05:
        r = 0.0
    return _clip_reward(r)


def reward_stand_still(
    command: np.ndarray, joint_angles: np.ndarray, default_pose: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """零命令时惩罚非默认姿态。"""
    if float(np.linalg.norm(command[:3])) >= threshold:
        return 0.0
    return _clip_reward(float(np.sum(np.abs(joint_angles - default_pose))))


def reward_abduction_angle(joint_angles: np.ndarray) -> float:
    """每条腿的 abduction（第 2 个关节，q[1::3]）尽量贴 0。"""
    return _clip_reward(float(np.sum(joint_angles[1::3] ** 2)))


# 全表，用于 Exercise 1 / Exercise 2。
REWARD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tracking_lin_vel":   {"sign": "+", "form": "exp(-‖v_xy − cmd_xy‖² / σ)",
                            "note": "指数跟踪，越接近指令越大"},
    "tracking_ang_vel":   {"sign": "+", "form": "exp(-(wz − cmd_wz)² / σ)",
                            "note": "偏航跟踪"},
    "tracking_orientation": {"sign": "+", "form": "exp(-‖z_b − z*_b‖² / σ)",
                              "note": "姿态跟踪（希望为正）"},
    "lin_vel_z":          {"sign": "−", "form": "v_z²",
                            "note": "惩罚垂直速度"},
    "ang_vel_xy":         {"sign": "−", "form": "‖ω_xy‖²",
                            "note": "惩罚 roll/pitch 角速度"},
    "orientation":        {"sign": "−", "form": "‖z_b[:2]‖²",
                            "note": "惩罚歪头"},
    "torques":            {"sign": "−", "form": "‖τ‖²",
                            "note": "能量效率"},
    "joint_acceleration": {"sign": "−", "form": "‖(qd−qd_prev)/dt‖²",
                            "note": "加速度平滑"},
    "mechanical_work":    {"sign": "−", "form": "Σ|τ·q̇|",
                            "note": "机械功，真・能耗代理"},
    "action_rate":        {"sign": "−", "form": "‖Δa‖²",
                            "note": "动作平滑"},
    "feet_air_time":      {"sign": "+", "form": "Σ(air_time − 0.1)·first_contact",
                            "note": "鼓励长迈步（非 0 指令才给）"},
    "stand_still":        {"sign": "−", "form": "Σ|q − q_default|·1[‖cmd‖<0.1]",
                            "note": "零命令时强制站好"},
    "abduction_angle":    {"sign": "−", "form": "Σq[1::3]²",
                            "note": "两侧外展接近 0，避免劈叉"},
}


# =============================================================================
# Section B — Policy JSON loader + Pure-NumPy forward pass
# =============================================================================

class RTNeuralPolicy:
    """解析 pupperv3-mjx export 出来的 RTNeural 风格 JSON policy。
    和 neural_controller_rtneural 在 Pupper 上跑的完全等价（单精度浮点 / 同样的层顺序）。

    JSON schema（和 test_policy.json / parkour_policy.json 一致）：
        use_imu (bool), control_orientation (bool),
        observation_history (int), action_scale (float),
        kp (float), kd (float),
        default_joint_pos (12,), joint_upper_limits (12,), joint_lower_limits (12,),
        in_shape = [None, in_dim],
        layers = [ {type:"dense", activation, shape:[None, out_dim], weights:[W, b]}, ... ]
    """

    ACT_FNS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "elu":    lambda x: np.where(x >= 0, x, np.expm1(np.minimum(x, 0))),   # expm1 safe for large -x
        "tanh":   np.tanh,
        "relu":   lambda x: np.maximum(x, 0.0),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
        "linear": lambda x: x,
        "softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True),
    }

    def __init__(self, path: pathlib.Path | str):
        path = pathlib.Path(path)
        with open(path) as f:
            d = json.load(f)
        self.path = path
        self.use_imu: bool = bool(d.get("use_imu", True))
        self.control_orientation: bool = bool(d.get("control_orientation", True))
        self.observation_history: int = int(d.get("observation_history", OBSERVATION_HISTORY))
        self.action_scale: float = float(d.get("action_scale", 1.0))
        self.kp: float = float(d.get("kp", NOMINAL_KP))
        self.kd: float = float(d.get("kd", NOMINAL_KD))
        self.default_joint_pos = np.asarray(d["default_joint_pos"], dtype=np.float32)
        self.joint_upper_limits = np.asarray(d["joint_upper_limits"], dtype=np.float32)
        self.joint_lower_limits = np.asarray(d["joint_lower_limits"], dtype=np.float32)
        self.in_dim: int = int(d["in_shape"][1])

        # 解析层：每个 layer = {type:"dense", activation, shape, weights:[W, b]}
        # RTNeural 导出后，W 的形状是 (in_dim, out_dim)，b 的形状是 (out_dim,)。
        self.layers: List[Dict[str, Any]] = []
        for L in d["layers"]:
            W = np.asarray(L["weights"][0], dtype=np.float32)
            b = np.asarray(L["weights"][1], dtype=np.float32)
            act = L.get("activation", "linear")
            self.layers.append({"W": W, "b": b, "act": act, "out_dim": int(L["shape"][1])})

        # 合法性校验
        assert self.layers[0]["W"].shape[0] == self.in_dim, (
            f"in_dim mismatch: json in_shape[1]={self.in_dim} but layer0 W.shape={self.layers[0]['W'].shape}"
        )
        for i in range(len(self.layers) - 1):
            a_out = self.layers[i]["out_dim"]
            b_in = self.layers[i + 1]["W"].shape[0]
            assert a_out == b_in, f"layer {i}→{i+1} dim mismatch: {a_out} vs {b_in}"

    @property
    def out_dim(self) -> int:
        return self.layers[-1]["out_dim"]

    def num_parameters(self) -> int:
        return int(sum(L["W"].size + L["b"].size for L in self.layers))

    def forward(self, obs_flat: np.ndarray) -> np.ndarray:
        """obs_flat: (in_dim,) float32。返回 (out_dim,) float32。"""
        x = np.asarray(obs_flat, dtype=np.float32)
        if x.shape != (self.in_dim,):
            raise ValueError(f"expected obs shape ({self.in_dim},), got {x.shape}")
        for L in self.layers:
            x = x @ L["W"] + L["b"]
            x = self.ACT_FNS[L["act"]](x)
        return x

    def forward_action_to_joints(self, obs_flat: np.ndarray) -> np.ndarray:
        """完整的控制管线：obs → action（tanh 头 ∈ [-1,1]）→ q_target = q_default + action_scale · action。"""
        a = self.forward(obs_flat)
        return self.default_joint_pos + self.action_scale * a

    def summary(self) -> str:
        lines = [f"RTNeuralPolicy({self.path.name})"]
        lines.append(f"  use_imu={self.use_imu}  control_orientation={self.control_orientation}"
                     f"  obs_history={self.observation_history}  action_scale={self.action_scale}")
        lines.append(f"  kp={self.kp}  kd={self.kd}")
        lines.append(f"  in_dim={self.in_dim}  (= {self.observation_history} × {self.in_dim // self.observation_history}/frame)")
        for i, L in enumerate(self.layers):
            lines.append(f"  layer[{i}]: dense {L['W'].shape[0]:>4} → {L['out_dim']:>4}  act={L['act']}  "
                         f"W={L['W'].shape}  b={L['b'].shape}")
        lines.append(f"  total params = {self.num_parameters():,}")
        return "\n".join(lines)


def load_policy(name_or_path: str) -> RTNeuralPolicy:
    """Accept either 'test' / 'parkour' / 'policies/xxx.json' / absolute path."""
    p = pathlib.Path(name_or_path)
    if p.exists():
        return RTNeuralPolicy(p)
    if (POLICY_DIR / f"{name_or_path}.json").exists():
        return RTNeuralPolicy(POLICY_DIR / f"{name_or_path}.json")
    if (POLICY_DIR / f"{name_or_path}_policy.json").exists():
        return RTNeuralPolicy(POLICY_DIR / f"{name_or_path}_policy.json")
    raise FileNotFoundError(f"policy not found: {name_or_path} "
                             f"(searched cwd and {POLICY_DIR})")


# =============================================================================
# Section C — Observation builder（用来做 viewer / 推理时构造 obs）
# =============================================================================

def build_obs_frame(
    *,
    local_ang_vel: np.ndarray,
    gravity_in_body: np.ndarray,
    command: np.ndarray,
    desired_world_z_in_body: np.ndarray,
    joint_pos: np.ndarray,
    last_action: np.ndarray,
    default_pose: np.ndarray = DEFAULT_POSE,
) -> np.ndarray:
    """把一帧 36 维 observation 按上游 environment.py:581 的 concat 顺序拼起来。"""
    frame = np.concatenate([
        np.asarray(local_ang_vel, dtype=np.float32).reshape(3),
        np.asarray(gravity_in_body, dtype=np.float32).reshape(3),
        np.asarray(command, dtype=np.float32).reshape(3),
        np.asarray(desired_world_z_in_body, dtype=np.float32).reshape(3),
        (np.asarray(joint_pos, dtype=np.float32).reshape(12)
         - np.asarray(default_pose, dtype=np.float32).reshape(12)),
        np.asarray(last_action, dtype=np.float32).reshape(12),
    ])
    return np.clip(frame, -100.0, 100.0).astype(np.float32)


def roll_obs_history(history: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
    """上游 jp.roll(hist, 36).at[:36].set(frame) —— newest 放最前面。"""
    assert history.shape[0] % new_frame.shape[0] == 0
    out = np.roll(history, new_frame.shape[0])
    out[: new_frame.shape[0]] = new_frame
    return out


# =============================================================================
# Section D — 辅助：从 MuJoCo sim 里抽一条随机 rollout（给 Exercise 2 / 8 用）
# =============================================================================

def simulate_random_rollout(
    n_steps: int = 100, seed: int = 0, action_noise: float = 0.3,
    apply_default_pose: bool = True,
) -> Dict[str, np.ndarray]:
    """浮动基 Pupper，PD（Kp=5, Kv=0.1）位置控制 + 随机扰动 action。
    返回一条轨迹（时长 = n_steps × 0.002s）的关节量 / 力矩 / base 姿态。
    用来做所有分析类 exercise 的"合成输入"。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_FLOATING))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    # 放到默认姿态附近。
    if apply_default_pose:
        data.qpos[0:3] = [0.0, 0.0, 0.20]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:19] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    q_log, qd_log, tau_log = [], [], []
    base_quat_log, base_linvel_log, base_angvel_log = [], [], []

    for _ in range(n_steps):
        # ctrl = default_pose + noise（模拟 policy output）
        data.ctrl[:] = DEFAULT_POSE + rng.normal(0, action_noise, 12)
        mujoco.mj_step(model, data)
        q_log.append(data.qpos[7:19].copy())
        qd_log.append(data.qvel[6:18].copy())
        tau_log.append(data.qfrc_actuator[6:18].copy())
        base_quat_log.append(data.xquat[base_id].copy())
        base_linvel_log.append(data.cvel[base_id, 3:6].copy())   # linear [vx,vy,vz] 世界系
        base_angvel_log.append(data.cvel[base_id, 0:3].copy())   # angular
    return {
        "q":       np.asarray(q_log),
        "qd":      np.asarray(qd_log),
        "tau":     np.asarray(tau_log),
        "base_quat":   np.asarray(base_quat_log),
        "base_linvel": np.asarray(base_linvel_log),
        "base_angvel": np.asarray(base_angvel_log),
        "dt": float(model.opt.timestep),
    }


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q=(w,x,y,z)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R @ v


def quat_inv_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return quat_rotate(np.array([w, -x, -y, -z]), v)
