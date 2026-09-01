"""Lab 2 TODO starter：Pupper 单腿 FK 与滑杆遥操作。

作者侧先运行：

    uv run python lab_2_fk_teleop/starter_todo.py

它应该停在第一个 NotImplementedError。交付学生版时，抽走已填完的
`starter.py`，再把本文件改名为 `starter.py`。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
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

from shared.kinematics import Transform  # noqa: E402
from shared.kinematics.pupper_leg import (  # noqa: E402
    T_BASE_HIP_FIXED,
    T_HIP_UPPER_FIXED,
    T_LOWER_FOOT,
    T_UPPER_LOWER_FIXED,
    T_WORLD_BASE,
)
from shared.viz import gif_utils, plot_utils  # noqa: E402


LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "models" / "scene.xml"
PORTFOLIO_DIR = LAB_DIR / "portfolio"

JOINT_NAMES = ("HAA", "HFE", "KFE")
ACTUATOR_NAMES = ("haa_motor", "hfe_motor", "kfe_motor")
FOOT_SITE_NAME = "foot"
JOINT_LIMITS = np.array(
    (
        (-2.51, 1.22),
        (-3.14, 0.42),
        (-0.71, 2.79),
    ),
    dtype=float,
)

GIF_SECONDS = 10.0
GIF_FPS = 15
FRAME_SIZE = (720, 400)
SCENE_TARGET_Y_PX = 215.0

REVERSE_DEMO_SECONDS = 1.5
SWEEP_SECONDS = GIF_SECONDS - REVERSE_DEMO_SECONDS
REVERSE_DEMO_THETA = np.array((0.4, -1.4, 1.6))

def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 Lab 2 的单腿 MJCF。"""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def _joint_qpos_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"缺少关节 {name!r}")
        ids.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(ids, dtype=int)


def _site_id(model: mujoco.MjModel, name: str = FOOT_SITE_NAME) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"缺少 site {name!r}")
    return int(site_id)


def _actuator_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    ids = []
    for name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"缺少执行器 {name!r}")
        ids.append(int(actuator_id))
    return tuple(ids)


def check_scene_unlocked(model: mujoco.MjModel) -> None:
    """任务 A：确认 Lab 1 的 `<equality>` 已删除，三关节和三 motor 都在。"""

    if model.neq != 0:
        raise AssertionError("scene.xml 里还存在 equality 约束，HAA/KFE 没有解锁")
    _joint_qpos_ids(model)
    _actuator_ids(model)


def fk_leg(theta: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """任务 B：返回 foot site 在世界坐标系下的位置。"""

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError("theta 必须是 (HAA, HFE, KFE) 三元组")
    haa, hfe, kfe = theta

    # TODO 1: 写出 T_world_HAA / T_HAA_HFE / T_HFE_KFE / T_KFE_foot，并连乘读第 4 列前三行。
    raise NotImplementedError("TODO 1: 写 Pupper 单腿 FK chain")


def fk_validate(
    *,
    seed: int = 0,
    n: int = 100,
    fk_fn=fk_leg,
) -> float:
    """任务 C：用 MuJoCo `mj_forward` 校验自写 FK，返回最大 foot 位置误差。"""

    model, data = load_model()
    check_scene_unlocked(model)
    qpos_ids = _joint_qpos_ids(model)
    foot_id = _site_id(model)
    rng = np.random.default_rng(seed)

    # TODO 2: 随机采样 n 组关节角，调用 mj_forward 取 foot site，与 fk_fn(theta) 对比并返回 max_err。
    raise NotImplementedError("TODO 2: 用 mj_forward 校验 FK")


def sample_workspace(n_samples: int = 20_000, *, seed: int = 0) -> np.ndarray:
    """任务 E：随机采样关节角，用 `fk_leg` 得到 foot 工作空间点云。"""

    if n_samples <= 0:
        raise ValueError("n_samples 必须为正数")
    rng = np.random.default_rng(seed)

    # TODO 3: 在 JOINT_LIMITS 内均匀采样关节角，并返回形状为 (n_samples, 3) 的 foot 位置数组。
    raise NotImplementedError("TODO 3: 采样工作空间")


def zero_pose_error() -> float:
    """零位时打印用的小自检。"""

    model, data = load_model()
    qpos_ids = _joint_qpos_ids(model)
    foot_id = _site_id(model)
    data.qpos[qpos_ids] = 0.0
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(fk_leg(np.zeros(3)) - data.site_xpos[foot_id]))


def save_workspace_plot(points: np.ndarray, path: Path) -> None:
    """把任务 E 的 3D 葫芦工作空间写成 PNG。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.0))
    plot_utils.workspace_scatter(
        fig,
        points,
        title=f"Pupper 单腿 foot 工作空间（N={len(points)}）",
        point_size=1.0,
        alpha=0.1,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scripted_joint_sweep(t: float) -> np.ndarray:
    """GIF 用的脚本轨迹：前 SWEEP_SECONDS 每段扫一个关节，最后一段定格做反例。"""

    if t >= SWEEP_SECONDS:
        return REVERSE_DEMO_THETA.copy()
    q = np.zeros(3)
    phase = t / (SWEEP_SECONDS / 3.0)
    joint_idx = min(int(phase), 2)
    local = phase - joint_idx
    lo, hi = JOINT_LIMITS[joint_idx]
    center = 0.5 * (lo + hi)
    amp = 0.42 * (hi - lo)
    q[joint_idx] = center + amp * math.sin(2.0 * math.pi * local)
    return q


def fk_leg_broken_demo(theta: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """反例：把 HFE 一段写成绕 x，让红球明显飞离 foot site。仅供 GIF 末尾演示。"""

    theta = np.asarray(theta, dtype=float)
    haa, hfe, kfe = theta
    T_world_HAA = T_WORLD_BASE @ T_BASE_HIP_FIXED @ Transform.rot_z(haa)
    T_HAA_HFE = T_HIP_UPPER_FIXED @ Transform.rot_x(hfe)
    T_HFE_KFE = T_UPPER_LOWER_FIXED @ Transform.rot_z(kfe)
    T_world_foot = T_world_HAA @ T_HAA_HFE @ T_HFE_KFE @ T_LOWER_FOOT
    return T_world_foot[:3, 3].copy()


def _add_marker(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    rgba=(1.0, 0.0, 0.0, 1.0),
    *,
    radius: float = 0.032,
) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _caption_frame(
    frame: np.ndarray,
    theta: np.ndarray,
    *,
    is_broken: bool = False,
) -> np.ndarray:
    panel_x = FRAME_SIZE[0] - 205
    scene = gif_utils.fit_scene_frame(
        frame,
        output_size=FRAME_SIZE,
        content_target=(0.5 * panel_x, SCENE_TARGET_Y_PX),
    )
    image = Image.fromarray(scene)
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(18)
    draw.rectangle((panel_x, 0, image.width, image.height), fill=(255, 255, 255, 218))
    if is_broken:
        lines = (
            "反例：HFE 误",
            "写成 rot_x",
            "",
            f"HAA = {theta[0]:+.2f} rad",
            f"HFE = {theta[1]:+.2f} rad",
            f"KFE = {theta[2]:+.2f} rad",
            "",
            "红球离开 foot",
        )
        text_color = (180, 0, 0, 255)
    else:
        lines = (
            "Pupper 单腿 FK",
            "",
            f"HAA = {theta[0]:+.2f} rad",
            f"HFE = {theta[1]:+.2f} rad",
            f"KFE = {theta[2]:+.2f} rad",
            "",
            "红球 = NumPy FK",
            "foot site 应贴住它",
        )
        text_color = (20, 20, 20, 255)
    y = 18
    for line in lines:
        draw.text((panel_x + 16, y), line, fill=text_color, font=font)
        y += 30
    return np.asarray(image)


def render_scripted_sweep(seconds: float = GIF_SECONDS, fps: int = GIF_FPS) -> list[np.ndarray]:
    """渲染 10 秒脚本扫关节 GIF 帧。"""

    model, data = load_model()
    qpos_ids = _joint_qpos_ids(model)
    renderer = mujoco.Renderer(model, height=FRAME_SIZE[1], width=FRAME_SIZE[0])
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(int(round(seconds * fps))):
            t = frame_idx / fps
            theta = scripted_joint_sweep(t)
            data.qpos[qpos_ids] = theta
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            is_broken = t >= SWEEP_SECONDS
            sphere_pos = fk_leg_broken_demo(theta) if is_broken else fk_leg(theta)
            _add_marker(renderer.scene, sphere_pos)
            frames.append(_caption_frame(renderer.render(), theta, is_broken=is_broken))
    finally:
        renderer.close()
    return frames


@dataclass
class TeleopState:
    """tkinter 线程和 MuJoCo viewer 主循环之间共享的三关节角。"""

    q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    running: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_joint(self, idx: int, value: float) -> None:
        with self.lock:
            self.q[idx] = value

    def get(self) -> np.ndarray:
        with self.lock:
            return self.q.copy()


def _run_tk_sliders(state: TeleopState) -> None:
    """三个滑杆单独跑在 tkinter 线程里，viewer 主线程只读数值。"""

    import tkinter as tk

    root = tk.Tk()
    root.title("Lab 2 Pupper FK teleop")

    def on_close() -> None:
        state.running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    for idx, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[idx]
        tk.Label(root, text=f"{name} [rad]").pack(anchor="w", padx=12)
        scale = tk.Scale(
            root,
            from_=float(lo),
            to=float(hi),
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=360,
            command=lambda value, i=idx: state.set_joint(i, float(value)),
        )
        scale.set(0.0)
        scale.pack(padx=12, pady=(0, 8))
    root.mainloop()


def launch_teleop_viewer() -> None:
    """任务 D：打开 viewer，用滑杆改 qpos，并叠红色 FK 小球。"""

    import mujoco.viewer

    model, data = load_model()
    qpos_ids = _joint_qpos_ids(model)
    state = TeleopState()
    ui_thread = threading.Thread(target=_run_tk_sliders, args=(state,), daemon=True)
    ui_thread.start()

    print("拖动 tkinter 滑杆控制 HAA/HFE/KFE；关闭 viewer 或滑杆窗口退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and state.running:
            theta = state.get()
            data.qpos[qpos_ids] = theta
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            _add_marker(viewer.user_scn, fk_leg(theta))
            viewer.sync()
            time.sleep(model.opt.timestep)
    state.running = False


def main() -> None:
    model, _ = load_model()
    check_scene_unlocked(model)
    err0 = zero_pose_error()
    max_err = fk_validate(seed=0, n=100)
    sample_workspace(500, seed=1)
    print(f"零位 (HAA, HFE, KFE) = (0, 0, 0) 时 |fk_leg - mj_forward| = {err0:.2e} m")
    print(f"100 组随机姿态 max_err = {max_err:.2e} m")


if __name__ == "__main__":
    if "--viewer" in sys.argv:
        launch_teleop_viewer()
    else:
        main()
