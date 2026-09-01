"""Lab 4 starter：Pupper URDF 手术 + PD 参数扫描。

本文件是教程主线使用的完整脚本。建议先直接运行，确认三只
Pupper 都能站住，再按 `make_variant` -> `find_stand_pose` ->
`find_stable_pd_gains` 的顺序阅读修改过程。
"""

from __future__ import annotations

import pickle
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.controllers.pd_controller import PDGains  # noqa: E402
from shared.viz import gif_utils, plot_utils  # noqa: E402


LAB_DIR = Path(__file__).resolve().parent
MODEL_DIR = LAB_DIR / "models"
PORTFOLIO_DIR = LAB_DIR / "portfolio"
SKELETON_PATH = EXERCISES_DIR / "shared" / "models" / "skeleton.xml"

LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_SUFFIXES = ("HAA", "HFE", "KFE")
THIGH_LEN = 0.08
CALF_LEN = 0.11
THIGH_MASS = 0.186
CALF_MASS = 0.050
HIP_MASS = 0.045
TORSO_MASS = 1.506
TORSO_HALF_SIZE = (0.10, 0.055, 0.025)
BALLAST_HALF_SIZE = (0.065, 0.042, 0.018)
FOOT_MASS = 0.020
FOOT_RADIUS = 0.018
DT = 0.004
MAX_TORQUE = 18.0
KP_GRID = np.array((10.0, 30.0, 60.0, 120.0), dtype=float)
KD_GRID = np.array((0.5, 1.0, 2.0, 5.0), dtype=float)
CAPTURE_FPS = 12
FRAME_SIZE = (1280, 480)
PANEL_SIZE = (426, 480)
SCENE_TARGET_Y_PX = 285.0


@dataclass(frozen=True)
class VariantSpec:
    key: str
    title: str
    file_name: str
    leg_scale: float
    torso_mass_scale: float


@dataclass
class StandTrace:
    time: np.ndarray
    base_z: np.ndarray
    q_error_rms: np.ndarray
    frames: list[np.ndarray] | None = None

    @property
    def last_second_z_std(self) -> float:
        mask = self.time >= self.time[-1] - 1.0
        return float(np.std(self.base_z[mask]))


@dataclass
class VariantResult:
    spec: VariantSpec
    path: Path
    stand_pose: np.ndarray
    base_height: float
    best_gains: PDGains
    z_std_grid: np.ndarray
    tuned_trace: StandTrace


VARIANTS = {
    "original": VariantSpec("original", "original", "pupper_v3.xml", 1.0, 1.0),
    "longleg": VariantSpec("longleg", "long-leg", "pupper_longleg.xml", 1.5, 1.0),
    "heavy": VariantSpec("heavy", "heavy", "pupper_heavy.xml", 1.0, 2.0),
}


def _variant_spec(name: str, *, leg_scale: float | None = None, torso_mass_scale: float | None = None) -> VariantSpec:
    key = name.lower().replace("_", "").replace("-", "")
    aliases = {"v3": "original", "pupperv3": "original", "long": "longleg", "longleg": "longleg"}
    key = aliases.get(key, key)
    base = VARIANTS.get(key)
    if base is None:
        file_key = key if key.startswith("pupper_") else f"pupper_{key}"
        base = VariantSpec(key, key, f"{file_key}.xml", 1.0, 1.0)
    return VariantSpec(
        base.key,
        base.title,
        base.file_name,
        float(base.leg_scale if leg_scale is None else leg_scale),
        float(base.torso_mass_scale if torso_mass_scale is None else torso_mass_scale),
    )


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _read_skeleton_template() -> ET.Element:
    """读取共享 skeleton，并确认它使用 variant_* default 注入点。"""

    root = ET.parse(SKELETON_PATH).getroot()
    names = {geom.attrib.get("class") for geom in root.findall(".//geom")}
    if "variant_thigh" not in names or "variant_calf" not in names:
        raise AssertionError("shared/models/skeleton.xml 缺少 variant_thigh / variant_calf 注入点")
    return root


def make_variant(name: str, *, leg_scale: float = 1.0, torso_mass_scale: float = 1.0) -> Path:
    """写出一份只含 default 注入 + skeleton include 的 Pupper 变体 MJCF。"""

    if leg_scale <= 0.0:
        raise ValueError("leg_scale 必须为正数")
    if torso_mass_scale <= 0.0:
        raise ValueError("torso_mass_scale 必须为正数")
    _read_skeleton_template()

    spec = _variant_spec(name, leg_scale=leg_scale, torso_mass_scale=torso_mass_scale)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / spec.file_name

    # 所有几何差异都从两个缩放量派生。这样 original / long-leg / heavy
    # 仍然共享同一份 skeleton.xml，只在 default class 上换参数。
    thigh_len = THIGH_LEN * spec.leg_scale
    calf_len = CALF_LEN * spec.leg_scale
    thigh_mass = THIGH_MASS * spec.leg_scale
    calf_mass = CALF_MASS * spec.leg_scale
    torso_mass = TORSO_MASS * spec.torso_mass_scale

    # heavy 背上的 ballast 是视觉提示，不参与质量计算；真正的质量变化
    # 已经写进 variant_torso 的 mass。这样不会把 torso 画成不自然的大盒子。
    show_ballast = spec.torso_mass_scale > 1.01
    ballast_size = BALLAST_HALF_SIZE if show_ballast else (0.001, 0.001, 0.001)
    ballast_rgba = "0.18 0.22 0.26 1" if show_ballast else "0 0 0 0"

    xml = f"""<mujoco model=\"{spec.key}\">
  <compiler angle=\"radian\" meshdir=\"../../shared/models/meshes/\" autolimits=\"true\"/>
  <default>
    <joint armature=\"0.003\" damping=\"0.04\" frictionloss=\"0.002\" limited=\"true\"/>
    <geom rgba=\"0.62 0.72 0.86 1\" contype=\"0\" conaffinity=\"0\"/>
    <default class=\"variant_torso\"><geom type=\"box\" size=\"{_fmt(TORSO_HALF_SIZE[0])} {_fmt(TORSO_HALF_SIZE[1])} {_fmt(TORSO_HALF_SIZE[2])}\" mass=\"{_fmt(torso_mass)}\" rgba=\"0.55 0.66 0.82 0.20\"/></default>
    <default class=\"variant_ballast\"><geom type=\"box\" size=\"{_fmt(ballast_size[0])} {_fmt(ballast_size[1])} {_fmt(ballast_size[2])}\" mass=\"0\" rgba=\"{ballast_rgba}\" contype=\"0\" conaffinity=\"0\"/></default>
    <default class=\"variant_hip\"><geom type=\"sphere\" size=\"0.026\" mass=\"{_fmt(HIP_MASS)}\" rgba=\"0.38 0.58 0.78 1\"/></default>
    <default class=\"variant_thigh\"><geom type=\"capsule\" fromto=\"0 0 0 0 0 -{_fmt(thigh_len)}\" size=\"0.013\" mass=\"{_fmt(thigh_mass)}\" rgba=\"0.52 0.62 0.82 1\"/></default>
    <default class=\"variant_calf\"><geom type=\"capsule\" fromto=\"0 0 0 0 0 -{_fmt(calf_len)}\" size=\"0.011\" mass=\"{_fmt(calf_mass)}\" rgba=\"0.50 0.78 0.56 1\"/></default>
    <default class=\"variant_foot\"><geom type=\"sphere\" pos=\"0 0 -{_fmt(calf_len)}\" size=\"{_fmt(FOOT_RADIUS)}\" mass=\"{_fmt(FOOT_MASS)}\" rgba=\"0.86 0.22 0.22 1\" contype=\"1\" conaffinity=\"1\" condim=\"3\" friction=\"1.2 0.04 0.01\"/><site pos=\"0 0 -{_fmt(calf_len)}\" size=\"0.015\" rgba=\"1 0.2 0.2 1\"/></default>
  </default>
  <include file=\"../../shared/models/skeleton.xml\"/>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")

    # 写完立即让 MuJoCo 编译一次，尽早发现 include 路径、class 名或
    # fromto 数字问题。后面的测试和渲染都建立在这一步能通过之上。
    mujoco.MjModel.from_xml_path(str(path))
    return path


def make_all_variants() -> dict[str, Path]:
    """写出三份变体 MJCF 和 zoo 场景。"""

    paths = {
        spec.key: make_variant(spec.key, leg_scale=spec.leg_scale, torso_mass_scale=spec.torso_mass_scale)
        for spec in VARIANTS.values()
    }
    write_zoo_scene()
    return paths


def write_zoo_scene() -> Path:
    """写出可编译的三 Pupper 场景，用 attach prefix 避免名字冲突。"""

    path = MODEL_DIR / "pupper_zoo.xml"
    # 这里不用连续 include 三份 XML，因为三份模型里都有 base/root/FL_HAA
    # 这些同名对象，直接 include 会撞名。MuJoCo 的 asset/model + attach
    # 可以在挂载时给每只机器人加 prefix，因此三只可以放在同一世界里观察。
    xml = """<mujoco model="pupper_zoo">
  <compiler angle="radian" meshdir="../../shared/models/meshes/" autolimits="true"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".2 .4 .6" rgb2=".4 .6 .8"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>
    <model name="original" file="pupper_v3.xml"/>
    <model name="longleg" file="pupper_longleg.xml"/>
    <model name="heavy" file="pupper_heavy.xml"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2.8 1.2 .05" material="grid" condim="3" contype="1" conaffinity="1"/>
    <light name="key" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <camera name="iso" pos="0.90 -1.15 0.68" xyaxes="0.787 0.616 0 -0.222 0.284 0.933"/>
    <frame pos="-0.55 0 0"><attach model="original" body="base" prefix="original/"/></frame>
    <frame pos="0 0 0"><attach model="longleg" body="base" prefix="longleg/"/></frame>
    <frame pos="0.55 0 0"><attach model="heavy" body="base" prefix="heavy/"/></frame>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(path))
    return path


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 MJCF，并统一仿真步长。"""

    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = DT
    return model, mujoco.MjData(model)


def _joint_names(prefix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{leg}_{suffix}" for leg in LEG_ORDER for suffix in JOINT_SUFFIXES)


def _site_names(prefix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{leg}_foot" for leg in LEG_ORDER)


def _joint_qpos_qvel_ids(model: mujoco.MjModel, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    for name in _joint_names(prefix):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"缺少关节 {name!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=int), np.asarray(qvel_ids, dtype=int)


def _base_qpos_qvel_addr(model: mujoco.MjModel, prefix: str = "") -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}root")
    if joint_id < 0:
        raise ValueError(f"缺少 freejoint {prefix}root")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _foot_geometry_lengths(model: mujoco.MjModel) -> tuple[float, float]:
    calf_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf")
    foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "FL_foot")
    if calf_body < 0 or foot_site < 0:
        raise ValueError("模型缺少 FL_calf 或 FL_foot")
    thigh_len = float(np.linalg.norm(model.body_pos[calf_body]))
    calf_len = float(np.linalg.norm(model.site_pos[foot_site]))
    return thigh_len, calf_len


def _leg_xz(hfe: float, kfe: float, thigh_len: float, calf_len: float) -> tuple[float, float]:
    x = -thigh_len * np.sin(hfe) - calf_len * np.sin(hfe + kfe)
    z = -thigh_len * np.cos(hfe) - calf_len * np.cos(hfe + kfe)
    return float(x), float(z)


def find_stand_pose(model: mujoco.MjModel, leg_scale: float = 1.0) -> np.ndarray:
    """搜索一组 HAA=0 的 12 维 stand_pose。"""

    thigh_len, calf_len = _foot_geometry_lengths(model)
    target_z = -0.14 * max(1.0, min(float(leg_scale), 1.5))
    hfe_grid = np.linspace(0.18, 1.20, 220)
    kfe_grid = np.linspace(-2.30, -0.25, 260)

    # 把单腿近似成 x-z 平面里的二连杆：HAA 固定为 0，只搜索 HFE/KFE。
    # 分数的三项分别约束足端高度、前后偏移和姿态自然度。
    best_score = np.inf
    best = (0.70, -1.40)
    for hfe in hfe_grid:
        for kfe in kfe_grid:
            x, z = _leg_xz(float(hfe), float(kfe), thigh_len, calf_len)
            score = (z - target_z) ** 2 + 0.25 * x**2 + 0.002 * (hfe - 0.70) ** 2
            if score < best_score:
                best_score = score
                best = (float(hfe), float(kfe))

    # 四条腿共用同一组局部关节角。后续如果要做不对称站姿，可以从这里
    # 改成按 FL/FR/RL/RR 分别搜索。
    haa_hfe_kfe = np.array((0.0, best[0], best[1]), dtype=float)
    return np.tile(haa_hfe_kfe, len(LEG_ORDER))


def _set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    stand_pose: np.ndarray,
    *,
    base_height: float,
    prefix: str = "",
    x_offset: float = 0.0,
    lifted: bool = True,
) -> None:
    mujoco.mj_resetData(model, data)
    base_qpos, _ = _base_qpos_qvel_addr(model, prefix)
    qpos_ids, _ = _joint_qpos_qvel_ids(model, prefix)
    data.qpos[base_qpos : base_qpos + 7] = np.array(
        [x_offset, 0.0, base_height + (0.012 if lifted else 0.0), 1.0, 0.0, 0.0, 0.0]
    )
    data.qpos[qpos_ids] = np.asarray(stand_pose, dtype=float)
    mujoco.mj_forward(model, data)


def base_height_for_pose(model: mujoco.MjModel, stand_pose: np.ndarray) -> float:
    """根据 foot site 的最低 z，给 free base 一个刚好离地的高度。"""

    data = mujoco.MjData(model)
    # 先把 base 放在 z=0，并套入关节站姿。此时 foot site 相对 base 的
    # 最低高度就是"机身需要抬多高才能让脚刚好接近地面"的依据。
    _set_initial_pose(model, data, stand_pose, base_height=0.0, lifted=False)
    z_values = []
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_z = float(data.xpos[base_id, 2])
    for site_name in _site_names():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        z_values.append(float(data.site_xpos[site_id, 2] - base_z))
    # FOOT_RADIUS 让球形 foot 的最低点贴地；额外 6 mm 给接触求解器一点余量，
    # 避免初始化时脚已经深插进地面。
    return float(-min(z_values) + FOOT_RADIUS + 0.006)


def _pd_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    stand_pose: np.ndarray,
    gains: PDGains,
    *,
    prefix: str = "",
) -> float:
    qpos_ids, qvel_ids = _joint_qpos_qvel_ids(model, prefix)

    # floating 模型的 qpos/qvel 前段属于 free base，不能进入关节 PD。
    # _joint_qpos_qvel_ids 通过关节名取地址，确保这里只拿到 12 个电机关节。
    q = data.qpos[qpos_ids]
    qdot = data.qvel[qvel_ids]
    tau = gains.kp * (stand_pose - q) + gains.kd * (0.0 - qdot)

    # MuJoCo actuator 顺序和 LEG_ORDER/JOINT_SUFFIXES 对齐。力矩先限幅，
    # 避免某些过硬参数在网格扫描时把仿真直接打飞。
    data.ctrl[:] = 0.0
    data.ctrl[: len(tau)] = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
    return float(np.sqrt(np.mean((stand_pose - q) ** 2)))


def simulate_stand(
    model: mujoco.MjModel,
    stand_pose: np.ndarray,
    gains: PDGains,
    *,
    seconds: float = 6.0,
    base_height: float | None = None,
    disturbance: bool = True,
    capture_frames: bool = False,
    label: str = "",
) -> StandTrace:
    """运行单只 Pupper 的站立闭环。"""

    data = mujoco.MjData(model)
    if base_height is None:
        base_height = base_height_for_pose(model, stand_pose)
    _set_initial_pose(model, data, stand_pose, base_height=base_height)

    # capture_frames 只在生成图片/GIF 时打开；纯数值扫描时关闭 renderer，
    # 否则 4x4x3 的 PD 网格会慢很多。
    renderer = mujoco.Renderer(model, height=360, width=420) if capture_frames else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "iso")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    ts: list[float] = []
    z_log: list[float] = []
    err_log: list[float] = []
    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    steps = int(round(seconds / DT))

    for step in range(steps):
        t = step * DT
        if disturbance:
            # 给 base 一个很小的竖直周期扰动。没有扰动时，很多偏软参数也能
            # "安静趴着"，不容易看出控制器余量。
            data.xfrc_applied[base_body_id, 2] = 2.0 * np.sin(2.0 * np.pi * 1.2 * t)
        err = _pd_step(model, data, stand_pose, gains)
        mujoco.mj_step(model, data)
        data.xfrc_applied[:, :] = 0.0

        ts.append(t)
        z_log.append(float(data.qpos[2]))
        err_log.append(err)

        if renderer is not None and t + 0.5 * DT >= next_frame_time:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frames.append(_caption_single_frame(frame, label=label))
            next_frame_time += 1.0 / CAPTURE_FPS

    return StandTrace(
        time=np.asarray(ts, dtype=float),
        base_z=np.asarray(z_log, dtype=float),
        q_error_rms=np.asarray(err_log, dtype=float),
        frames=frames if capture_frames else None,
    )


def find_stable_pd_gains(
    model: mujoco.MjModel,
    stand_pose: np.ndarray,
    kp_grid: np.ndarray = KP_GRID,
    kd_grid: np.ndarray = KD_GRID,
) -> tuple[float, float, np.ndarray]:
    """扫描 `(Kp, Kd)` 网格，返回当前网格里较稳定的一组 PD 参数。"""

    kp_grid = np.asarray(kp_grid, dtype=float)
    kd_grid = np.asarray(kd_grid, dtype=float)
    z_std = np.zeros((len(kd_grid), len(kp_grid)), dtype=float)
    base_height = base_height_for_pose(model, stand_pose)

    # 每个格子都用同一套 stand_pose 和 base_height，唯一变化是 Kp/Kd。
    # 指标取最后 1 秒 base z 标准差，避免把"趴得很稳"误判为站稳。
    for i, kd in enumerate(kd_grid):
        for j, kp in enumerate(kp_grid):
            trace = simulate_stand(
                model,
                stand_pose,
                PDGains(kp=float(kp), kd=float(kd)),
                seconds=6.0,
                base_height=base_height,
                disturbance=True,
                capture_frames=False,
            )
            z_std[i, j] = trace.last_second_z_std

    # 轻微惩罚过软的格子。它们有时 z_std 不差，但抗扰动余量很小，
    # 不适合作为后续 gait 的默认控制参数。
    score = z_std.copy()
    score += (kp_grid[None, :] < 30.0) * 1e-4
    score += (kd_grid[:, None] < 1.0) * 1e-4
    best_i, best_j = np.unravel_index(int(np.argmin(score)), score.shape)
    return float(kp_grid[best_j]), float(kd_grid[best_i]), z_std


def run_pd_sweeps() -> dict[str, VariantResult]:
    """生成变体、求站姿、扫描 PD，并返回三只 Pupper 的结果。"""

    paths = make_all_variants()
    results: dict[str, VariantResult] = {}
    for key, spec in VARIANTS.items():
        # 每只变体都单独经历同一套流程。不要把 original 的 stand_pose
        # 或 Kp/Kd 直接套给 long-leg/heavy，这正是本节想展示的连锁效应。
        model, _ = load_model(paths[key])
        stand_pose = find_stand_pose(model, spec.leg_scale)
        base_height = base_height_for_pose(model, stand_pose)
        best_kp, best_kd, grid = find_stable_pd_gains(model, stand_pose, KP_GRID, KD_GRID)
        gains = PDGains(kp=best_kp, kd=best_kd)
        tuned = simulate_stand(
            model,
            stand_pose,
            gains,
            seconds=10.0,
            base_height=base_height,
            disturbance=True,
            capture_frames=False,
        )
        results[key] = VariantResult(spec, paths[key], stand_pose, base_height, gains, grid, tuned)
    return results


def render_zoo_still(results: dict[str, VariantResult]) -> np.ndarray:
    """任务 4：渲染三只 Pupper 的最终站姿静帧。"""

    panels: list[np.ndarray] = []
    for key in ("original", "longleg", "heavy"):
        result = results[key]
        model, _ = load_model(result.path)
        data = mujoco.MjData(model)
        _set_initial_pose(model, data, result.stand_pose, base_height=result.base_height)
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for step in range(int(round(3.0 / DT))):
            t = step * DT
            data.xfrc_applied[base_body_id, 2] = 0.6 * np.sin(2.0 * np.pi * 1.2 * t)
            _pd_step(model, data, result.stand_pose, result.best_gains)
            mujoco.mj_step(model, data)
            data.xfrc_applied[:, :] = 0.0

        renderer = mujoco.Renderer(model, height=360, width=420)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "iso")
        renderer.update_scene(data, camera=camera)
        panels.append(_caption_single_frame(renderer.render(), label=result.spec.title))
    return np.concatenate(panels, axis=1)


def save_zoo_still(results: dict[str, VariantResult], path: Path) -> None:
    """把三只 Pupper 的最终站姿并排写成 PNG。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_zoo_still(results)).save(path)


def _caption_single_frame(frame: np.ndarray, *, label: str) -> np.ndarray:
    scene = gif_utils.fit_scene_frame(
        frame,
        output_size=PANEL_SIZE,
        content_target=(0.5 * PANEL_SIZE[0], SCENE_TARGET_Y_PX),
        background_rgb=(0, 0, 0),
    )
    image = Image.fromarray(scene)
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(22)
    draw.rectangle((0, 0, image.width, 44), fill=(255, 255, 255, 225))
    draw.text((16, 10), label, fill=(20, 40, 60, 255), font=font)
    return np.asarray(image)


def save_heatmap(results: dict[str, VariantResult], path: Path) -> None:
    """保存三只 Pupper 的 PD 参数扫描 heatmap。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_utils.apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    im = None
    for ax, key in zip(axes, ("original", "longleg", "heavy")):
        result = results[key]
        im = plot_utils.pd_heatmap(
            fig,
            KP_GRID,
            KD_GRID,
            result.z_std_grid,
            ax=ax,
            title=f"{result.spec.title}: best=({result.best_gains.kp:g}, {result.best_gains.kd:g})",
        )
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="最后 1 秒 base z 标准差 [mm]")
    fig.suptitle("Lab 4：三只 Pupper 的 PD 参数扫描")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stand_z_plot(results: dict[str, VariantResult], path: Path) -> None:
    """Stretch：把调好 PD 后的 base z 时间序列叠在一张图上。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_utils.apply_theme()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for key in ("original", "longleg", "heavy"):
        result = results[key]
        ax.plot(result.tuned_trace.time, result.tuned_trace.base_z, label=result.spec.title)
    ax.set_title("三只 Pupper 调好 PD 后的 base z")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("base z [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sweep_pickle(results: dict[str, VariantResult], path: Path) -> None:
    """把 16 格 heatmap 数据 pickle 到 portfolio，方便复查。"""

    payload = {
        key: {
            "kp_grid": KP_GRID,
            "kd_grid": KD_GRID,
            "z_std_grid": result.z_std_grid,
            "best": (result.best_gains.kp, result.best_gains.kd),
            "stand_pose": result.stand_pose,
        }
        for key, result in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def run_experiment() -> dict[str, VariantResult]:
    """make_artifacts.py 的主入口。"""

    return run_pd_sweeps()


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment()
    save_heatmap(results, PORTFOLIO_DIR / "pd_heatmap.png")
    save_zoo_still(results, PORTFOLIO_DIR / "pupper_zoo.png")
    save_stand_z_plot(results, PORTFOLIO_DIR / "stand_z_vs_t.png")
    save_sweep_pickle(results, PORTFOLIO_DIR / "pd_stable_gains.pkl")

    print(f"Lab 4 观察图已写入 {PORTFOLIO_DIR}/")
    for key in ("original", "longleg", "heavy"):
        result = results[key]
        print(
            f"{result.spec.title}: Kp={result.best_gains.kp:g}, Kd={result.best_gains.kd:g}, "
            f"z_std={result.tuned_trace.last_second_z_std * 1000:.2f} mm"
        )


if __name__ == "__main__":
    main()
