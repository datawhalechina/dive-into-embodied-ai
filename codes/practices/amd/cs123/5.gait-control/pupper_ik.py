"""真实 Pupper v3 的单腿数值 IK / FK（不依赖 shared.kinematics）。

§5 步态 demo 直接跑公共 mesh 模型 `assets/mjcfs/pupper_v3.xml`。真实模型的关节
是「固定 body 四元数 + 局部 z 轴 hinge」，零角=屈膝站姿，左右腿镜像——解析
玩具 IK 无法套用。这里用 MuJoCo 自身求解：设关节角 → `mj_forward` → 读
`foot_site` → `mj_jacSite` 取雅可比 → DLS 迭代。模型精确，四条腿自动正确，
脚一定落在真网格上，从根上避免 mesh 与 IK 的错位。
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np


MODEL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "assets"
    / "mjcfs"
    / "pupper_v3.xml"
)

# 对外统一用 FL/FR/RL/RR + HAA/HFE/KFE；映射到真实模型的 body/joint/site 命名。
LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_PREFIX = {
    "FL": "leg_front_l",
    "FR": "leg_front_r",
    "RL": "leg_back_l",
    "RR": "leg_back_r",
}
# 各腿髋关节（_1 body）在 base_link 系的位置，用于把 hip-local 足端轨迹搬到 base 系。
HIP_OFFSETS = {
    "FL": np.array((0.075, 0.0835, 0.0)),
    "FR": np.array((0.075, -0.0835, 0.0)),
    "RL": np.array((-0.075, 0.0725, 0.0)),
    "RR": np.array((-0.075, -0.0725, 0.0)),
}


def joint_names(leg: str) -> tuple[str, str, str]:
    """(HAA, HFE, KFE) 三个 hinge 关节名。"""
    p = LEG_PREFIX[leg]
    return (f"{p}_1", f"{p}_2", f"{p}_3")


def foot_site_name(leg: str) -> str:
    return f"{LEG_PREFIX[leg]}_3_foot_site"


class PupperLegIK:
    """按腿求解真实 Pupper 的 (HAA, HFE, KFE)。

    自带一份内部 MjData 做 IK：把 base 放到原点+单位姿态，于是 site 的世界坐标
    就等于 base 系坐标，DLS 直接在世界系迭代即可。求解只用到目标腿的 3 个 DOF，
    与仿真主循环互不干扰。
    """

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._data = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))

        m = self.model
        self.qadr: dict[str, list[int]] = {}
        self.dadr: dict[str, list[int]] = {}
        self.limits: dict[str, np.ndarray] = {}
        self.site: dict[str, int] = {}
        self.ctrl: dict[str, list[int]] = {}
        for leg in LEG_ORDER:
            jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names(leg)]
            aids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joint_names(leg)]
            if min(jids) < 0 or min(aids) < 0:
                raise ValueError(f"模型里缺少 {leg} 的关节/驱动器：{joint_names(leg)}")
            self.qadr[leg] = [int(m.jnt_qposadr[j]) for j in jids]
            self.dadr[leg] = [int(m.jnt_dofadr[j]) for j in jids]
            self.limits[leg] = np.array([m.jnt_range[j] for j in jids], dtype=float)
            self.ctrl[leg] = [int(a) for a in aids]
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, foot_site_name(leg))
            if sid < 0:
                raise ValueError(f"模型里缺少 {leg} 的 foot site：{foot_site_name(leg)}")
            self.site[leg] = sid

    def _foot(self, q: np.ndarray, leg: str) -> np.ndarray:
        d = self._data
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0  # base 单位四元数 → 世界系即 base 系
        d.qpos[self.qadr[leg]] = q
        mujoco.mj_forward(self.model, d)
        return d.site_xpos[self.site[leg]].copy()

    def fk(self, q: np.ndarray, *, leg: str) -> np.ndarray:
        """(HAA, HFE, KFE) → foot site 在 base 系的 (x, y, z)。"""
        q = np.asarray(q, dtype=float)
        if q.shape != (3,):
            raise ValueError("q 必须是 (HAA, HFE, KFE)")
        return self._foot(q, leg)

    def ik(
        self,
        foot_xyz,
        *,
        leg: str,
        q_seed: np.ndarray | None = None,
        tol: float = 1e-4,
        max_iter: int = 40,
        damping: float = 1e-3,
        step: float = 0.8,
    ) -> np.ndarray:
        """base 系足端目标 → (HAA, HFE, KFE)，DLS 迭代，带关节限位裁剪。"""
        target = np.asarray(foot_xyz, dtype=float)
        if target.shape != (3,):
            raise ValueError("foot_xyz 必须是长度为 3 的向量")
        lo, hi = self.limits[leg][:, 0], self.limits[leg][:, 1]
        q = np.zeros(3) if q_seed is None else np.asarray(q_seed, dtype=float).copy()
        q = np.clip(q, lo, hi)
        err = target - self._foot(q, leg)
        for _ in range(max_iter):
            if float(np.linalg.norm(err)) < tol:
                break
            mujoco.mj_jacSite(self.model, self._data, self._jacp, None, self.site[leg])
            J = self._jacp[:, self.dadr[leg]]
            delta = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(3), err)
            q = np.clip(q + step * delta, lo, hi)
            err = target - self._foot(q, leg)
        return q


def build_sim_model(
    *,
    stand_height: float,
    weld: bool = True,
    kp: float = 20.0,
    kd: float = 0.6,
) -> mujoco.MjModel:
    """从本地 XML 编译仿真模型：可选把 base_link 焊到世界、可选调硬位置伺服。

    weld 用 MjSpec 在运行时加，避免另写一份 XML，`meshdir` 保持正确。焊接点
    在 (0, 0, stand_height)、单位姿态；前进 trot 时逐帧改 `eq_data[3]`（relpose.x）。
    位置伺服默认 gainprm=5 偏软，步态目标变化快时跟踪滞后，这里调硬到 kp/kd。
    """
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

    for act in spec.actuators:
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kd

    if weld:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = "world"
        eq.name2 = "base_link"
        # data 布局：anchor(3) + relpose 位置(3) + relpose 四元数(4) + torquescale(1)
        eq.data[:] = np.array([0, 0, 0, 0, 0, stand_height, 1, 0, 0, 0, 1.0])
        eq.solref[:] = [0.002, 1.0]

    return spec.compile()


# eq_data 里 relpose 位置 x 的下标（前进 trot 平移焊接点用）。
WELD_RELPOSE_X = 3
