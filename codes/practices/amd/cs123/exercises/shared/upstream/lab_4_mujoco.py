"""
CS123 Lab 4 — MuJoCo 版（Layer 1：批量跑 + 画图）
模型化控制与小跑步态（Trotting gait），整机 Pupper V3 共 4 条腿 12 关节。

运行方式：
    uv run python lab_4_mujoco.py

本实验七个练习（1 个 FK 验证 + 1 个对称性 + 2 个步态几何 + 1 个缓存 + 2 个交付物分析）：

    Exercise 1 — 四条腿 FK vs. MuJoCo 真值（全部腿的脚尖位置都对到 ~µm）
    Exercise 2 — 左右对称性：FR↔FL / BR↔BL 的 y 分量互为相反数
    Exercise 3 — 六点梯形步态几何（touchdown/stance×3/liftoff/mid-swing）
    Exercise 4 — 相位图（gait diagram）：trot vs pace vs bound vs pronk
    Exercise 5 — cache_target_joint_positions：整个周期的 12 关节 + 4 脚尖时序
    Exercise 6 — 步态参数 → 前进速度（交付物 Part 6 / Part 7：race）
    Exercise 7 — COM / ee_offset 对静态稳定裕度的影响（交付物 Part 6）

单位约定（SI）：
    关节角 θᵢ                     [rad]
    连杆长度 / 脚尖位置 (x,y,z)    [m]
    代价（L2² 误差）               [m²]
    时间 t                         [s]

四条腿 FK 的推导（从 MJCF 的 body 四元数反推）：
    FR: T_0_1 = trans( 0.075, -0.0835, 0) @ R_x(+π/2) @ R_z(θ1)
    FL: T_0_1 = trans( 0.075, +0.0835, 0) @ R_x(-π/2) @ R_z(θ1)
    BR: T_0_1 = trans(-0.075, -0.0725, 0) @ R_x(+π/2) @ R_z(θ1)
    BL: T_0_1 = trans(-0.075, +0.0725, 0) @ R_x(-π/2) @ R_z(θ1)
其余两段对 R / L 有固定的 y 翻号（详见下面 _leg_fk_impl）。
"""

import pathlib
import time

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# 让 matplotlib 能画中文标题；和 lab3 一样用 PingFang/Noto 字体。
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
    "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

_DIR = pathlib.Path(__file__).resolve().parent
_SHARED_DIR = _DIR.parent
MODEL_FIXED = _SHARED_DIR / "models" / "pupper_v3_fixed.xml"
PLOT_DIR = _DIR / "tmp" / "plots"

# 12 关节顺序（和 MJCF / ros2_control yaml 一致）。
JOINT_ORDER = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1",  "leg_back_r_2",  "leg_back_r_3",
    "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
]
LEG_NAMES = ["fr", "fl", "br", "bl"]
LEG_SITES = {  # MuJoCo site name for each leg's foot tip
    "fr": "leg_front_r_3_foot_site",
    "fl": "leg_front_l_3_foot_site",
    "br": "leg_back_r_3_foot_site",
    "bl": "leg_back_l_3_foot_site",
}

# hip 相对于 base_link 的位置 [m] —— 直接来自 MJCF 的 leg_*_1 pos 字段。
HIP_OFFSETS = {
    "fr": np.array([ 0.075, -0.0835, 0.0]),
    "fl": np.array([ 0.075,  0.0835, 0.0]),
    "br": np.array([-0.075, -0.0725, 0.0]),
    "bl": np.array([-0.075,  0.0725, 0.0]),
}

# 上游 lab_4.py 里的"腿相对 base_link 的 offset"（脚尖轨迹用的，不是 hip 真实位置）。
# 它故意不等于 HIP_OFFSETS：把脚尖朝外/朝后挪一点，就是步态"站位"的调节旋钮。
LEG_EE_OFFSETS = {
    "fr": np.array([ 0.06, -0.09, 0.0]),
    "fl": np.array([ 0.06,  0.09, 0.0]),
    "br": np.array([-0.11, -0.09, 0.0]),
    "bl": np.array([-0.11,  0.09, 0.0]),
}

# 关节限位（URDF / MJCF 原值）。注意左右腿是镜像相反的。
JOINT_LIMITS = {
    "fr": [(-1.22, 2.51), (-0.42, 3.14), (-2.79, 0.71)],
    "fl": [(-2.51, 1.22), (-3.14, 0.42), (-0.71, 2.79)],
    "br": [(-1.22, 2.51), (-0.42, 3.14), (-2.79, 0.71)],
    "bl": [(-2.51, 1.22), (-3.14, 0.42), (-0.71, 2.79)],
}

# IK / PD 超参数（MJCF 用 Kp=5, Kv=0.1；IK 学习率沿用 lab3 的 10）。
DEFAULT_LR = 10.0
DEFAULT_MAX_ITER = 50
DEFAULT_TOL = 1e-4
DEFAULT_EPSILON = 1e-3

# -----------------------------------------------------------------------------
# 六点梯形步态（和上游 lab_4.py 的六个 position 对齐）。单位 [m]，hip 局部坐标。
# 轨迹顺序：touchdown → stand_1 → stand_2 → stand_3 → liftoff → mid_swing → 回到 touchdown。
# 前 5 段（4 条边）都贴地，是 stance；后 2 段（2 条边）是空中 swing。
# 所以 duty factor ≈ 4/6 ≈ 0.67 —— 实际是 trot-walk 混合体，保守一点不容易摔。
# -----------------------------------------------------------------------------
GAIT_KEY_POINTS = np.array([
    [ 0.05,  0.0, -0.14],   # 0 touchdown
    [ 0.025, 0.0, -0.14],   # 1 stand-1
    [ 0.0,   0.0, -0.14],   # 2 stand-2
    [-0.025, 0.0, -0.14],   # 3 stand-3
    [-0.05,  0.0, -0.14],   # 4 liftoff
    [ 0.0,   0.0, -0.05],   # 5 mid-swing
])
GAIT_PERIOD = 1.0   # [s]  一个完整步态周期
GAIT_N = len(GAIT_KEY_POINTS)  # 6

# 不同步态 → 4 条腿的相位偏移（单位是一个周期的分数）。
# 行顺序：fr, fl, br, bl。
GAIT_PATTERNS = {
    # 对角腿同相：(fr, bl) 一组；(fl, br) 一组，差半周期。
    "trot":  np.array([0.0, 0.5, 0.5, 0.0]),
    # 同侧腿同相：(fr, br) 和 (fl, bl)。像骆驼。
    "pace":  np.array([0.0, 0.5, 0.0, 0.5]),
    # 前后腿同相：(fr, fl) 和 (br, bl)。像兔子 / 袋鼠。
    "bound": np.array([0.0, 0.0, 0.5, 0.5]),
    # 四脚同时：所有腿同相。跳跃。
    "pronk": np.array([0.0, 0.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# 基础工具：4×4 齐次变换（和 lab2/lab3 完全一样）
# ---------------------------------------------------------------------------

def rotation_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])


def rotation_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])


def rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])


def translation(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


# ---------------------------------------------------------------------------
# 四条腿的正运动学（FK）——全部返回 base_link 坐标系下的脚尖位置 [m]
#
# 从 MJCF 的 body pos / quat 反推而来（没有走"抄一份 lab3 再猜 L 侧"的捷径）：
#   - 右腿（R）：T_0_1 里 R_x(+π/2)，T_2_3 平移 y=-0.0494，T_2_3 带 R_y(+π/2)；
#                foot_offset.y = -0.06216。
#   - 左腿（L）：T_0_1 里 R_x(-π/2)，T_2_3 平移 y=+0.0494，T_2_3 带 R_y(+π/2)；
#                foot_offset.y = +0.06216。
#   - 前腿：T_0_1 平移 (+0.075, ±0.0835, 0)。
#   - 后腿：T_0_1 平移 (-0.075, ±0.0725, 0)。
#   - T_1_2 四条腿都是 R_y(-π/2) @ R_z(θ2)，没有差异。
# ---------------------------------------------------------------------------

def _leg_fk_impl(theta1, theta2, theta3, hip_offset, side):
    """side ∈ {'R', 'L'}；hip_offset = base→hip_1_origin 的三维向量 [m]。"""
    sign = -1.0 if side == "R" else 1.0
    alpha = +np.pi / 2 if side == "R" else -np.pi / 2
    # Side-L 的 T_2_3 和 T_3_ee 的 y 方向整体翻一号；R_y 角度不变（都是 +π/2）。
    T_0_1 = translation(*hip_offset) @ rotation_x(alpha) @ rotation_z(theta1)
    T_1_2 = rotation_y(-np.pi / 2) @ rotation_z(theta2)
    T_2_3 = translation(0.0, sign * 0.0494, 0.0685) @ rotation_y(+np.pi / 2) @ rotation_z(theta3)
    T_3_ee = translation(0.06231, sign * 0.06216, 0.018)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


def fr_leg_fk(theta):
    return _leg_fk_impl(theta[0], theta[1], theta[2], HIP_OFFSETS["fr"], "R")


def fl_leg_fk(theta):
    return _leg_fk_impl(theta[0], theta[1], theta[2], HIP_OFFSETS["fl"], "L")


def br_leg_fk(theta):
    return _leg_fk_impl(theta[0], theta[1], theta[2], HIP_OFFSETS["br"], "R")


def bl_leg_fk(theta):
    return _leg_fk_impl(theta[0], theta[1], theta[2], HIP_OFFSETS["bl"], "L")


LEG_FK = {"fr": fr_leg_fk, "fl": fl_leg_fk, "br": br_leg_fk, "bl": bl_leg_fk}


def forward_kinematics(theta12):
    """12 维关节角 → (4, 3) 的四条腿脚尖位置（base_link 坐标）。"""
    th = np.asarray(theta12).reshape(4, 3)
    return np.stack([LEG_FK[name](th[i]) for i, name in enumerate(LEG_NAMES)], axis=0)


# ---------------------------------------------------------------------------
# 共用的 IK 内核（从 lab3 搬过来，保持每个 lab 自包含）。
# 代价 C(θ) = ||fk(θ) - target||²（squared L2，单位 m²），方便梯度下降。
# ---------------------------------------------------------------------------

def _cost(theta, target_ee, leg_fk):
    p = leg_fk(theta)
    e = p - target_ee
    return float(e @ e)


def _numerical_gradient(theta, target_ee, leg_fk, epsilon=DEFAULT_EPSILON):
    g = np.zeros(3)
    for i in range(3):
        tp = theta.copy(); tp[i] += epsilon
        tm = theta.copy(); tm[i] -= epsilon
        g[i] = (_cost(tp, target_ee, leg_fk) - _cost(tm, target_ee, leg_fk)) / (2 * epsilon)
    return g


def inverse_kinematics_single_leg(target_ee, leg_name, initial_guess=(0.0, 0.0, 0.0),
                                  lr=DEFAULT_LR, max_iter=DEFAULT_MAX_ITER,
                                  tol=DEFAULT_TOL, record=False):
    leg_fk = LEG_FK[leg_name]
    theta = np.array(initial_guess, dtype=float)
    hist = []
    for _ in range(max_iter):
        g = _numerical_gradient(theta, target_ee, leg_fk)
        theta = theta - lr * g
        if record:
            hist.append(_cost(theta, target_ee, leg_fk))
        l1_mean = float(np.mean(np.abs(leg_fk(theta) - target_ee)))
        if l1_mean < tol:
            break
    return theta, hist


# ---------------------------------------------------------------------------
# 步态工具
# ---------------------------------------------------------------------------

def leg_key_points(leg_name, ee_offset=None):
    """返回 (6, 3) 六个参考点（base_link 坐标系）。"""
    if ee_offset is None:
        ee_offset = LEG_EE_OFFSETS[leg_name]
    return GAIT_KEY_POINTS + ee_offset


def interpolate_gait(t, leg_name, pattern="trot", ee_offset=None, period=GAIT_PERIOD):
    """在时刻 t 查询某条腿的目标脚尖位置（base_link 坐标系，[m]）。

    - 沿 6 个关键点做分段线性插值，每段 1/6 个周期。
    - 不同腿通过 GAIT_PATTERNS[pattern] 给出的相位偏移错开（trot 是 0/0.5/0.5/0）。
    """
    pts = leg_key_points(leg_name, ee_offset)
    phase_shift = GAIT_PATTERNS[pattern][LEG_NAMES.index(leg_name)]
    phase = ((t / period) + phase_shift) % 1.0  # [0, 1)
    seg_len = 1.0 / GAIT_N
    seg = int(phase // seg_len)
    frac = (phase - seg * seg_len) / seg_len
    p0 = pts[seg]
    p1 = pts[(seg + 1) % GAIT_N]
    return (1 - frac) * p0 + frac * p1


def leg_phase_state(t, leg_name, pattern="trot", period=GAIT_PERIOD):
    """返回这条腿当前处于 stance 还是 swing（字符串）。
    swing 段定义为从 liftoff(4) → mid_swing(5) → touchdown(6=0)，占周期的 2/6。"""
    phase_shift = GAIT_PATTERNS[pattern][LEG_NAMES.index(leg_name)]
    phase = ((t / period) + phase_shift) % 1.0
    seg = int(phase // (1.0 / GAIT_N))
    return "swing" if seg in (4, 5) else "stance"


def cache_target_joint_positions(pattern="trot", period=GAIT_PERIOD, dt=0.02,
                                 ee_offsets=None):
    """预计算一个周期的 (N, 12) target_joints 和 (N, 4, 3) target_ee 缓存。

    和上游 lab_4.py 的 `cache_target_joint_positions` 等价，只是把结果形状改得更
    好用（四条腿分开存）。warm-start：每次 IK 用上一次的解做 initial_guess。
    """
    if ee_offsets is None:
        ee_offsets = LEG_EE_OFFSETS
    ts = np.arange(0.0, period, dt)
    n = len(ts)
    target_joints = np.zeros((n, 12))
    target_ee = np.zeros((n, 4, 3))

    # warm start 初值：左右腿的"home"姿态不同（θ₁ 方向相反），避免一开始就走到约束外。
    # R 侧 θ₁ ≥ 0 折腿，L 侧 θ₁ ≤ 0 折腿；θ₂ / θ₃ 是对角腿的关节范围差别，这里取中性初值 0。
    init_R = np.array([0.0, 0.0, 0.0])
    init_L = np.array([0.0, 0.0, 0.0])
    current = {name: (init_R if name in ("fr", "br") else init_L).copy() for name in LEG_NAMES}

    for k, t in enumerate(ts):
        for i, leg in enumerate(LEG_NAMES):
            tgt = interpolate_gait(t, leg, pattern=pattern, ee_offset=ee_offsets[leg], period=period)
            # 对 cache 稍微放松：更多迭代 + 和上游 lab4 对齐的容差 1e-3 m。
            th, _ = inverse_kinematics_single_leg(tgt, leg, initial_guess=current[leg],
                                                   max_iter=300, tol=1e-3)
            current[leg] = th
            target_joints[k, 3 * i : 3 * i + 3] = th
            target_ee[k, i] = tgt
    return target_joints, target_ee


# ---------------------------------------------------------------------------
# MuJoCo 真值工具（Exercise 1 / 5 共用）
# ---------------------------------------------------------------------------

def _foot_positions_from_mujoco(model, data, theta12):
    """把 12 维关节角写进 qpos，跑 mj_forward，读 4 条腿 site_xpos 回 base_link 坐标。
    fixed-base 版的 qpos 就是 12 个关节角。floating-base 版 qpos 前 7 个是 free joint，
    本函数只为 fixed-base 用。"""
    assert model.nq == 12, "foot_positions_from_mujoco 仅适用 fixed-base pupper"
    data.qpos[:] = theta12
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_pos = data.xpos[base_id].copy()
    base_R = data.xmat[base_id].reshape(3, 3).copy()
    feet = np.zeros((4, 3))
    for i, leg in enumerate(LEG_NAMES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEG_SITES[leg])
        p_world = data.site_xpos[sid]
        feet[i] = base_R.T @ (p_world - base_pos)
    return feet
