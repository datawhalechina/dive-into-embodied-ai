"""Lab 3 TODO starter：Pupper 单腿 Raibert 悬空踏步。

作者侧先运行：

    uv run python lab_3_stepping/starter_todo.py

它应该停在第一个 NotImplementedError。交付学生版时，抽走已填完的
`starter.py`，再把本文件改名为 `starter.py`。
"""

from __future__ import annotations

import sys
import time
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

from lab_2_fk_teleop.starter import _add_marker  # noqa: E402
from shared.controllers.pd_controller import PDGains  # noqa: E402
from shared.kinematics import damped_pinv, fk_leg, pose_error  # noqa: E402
from shared.kinematics.fk import T_WORLD_BASE  # noqa: E402
from shared.trajectories import stance_swing_phase  # noqa: E402
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

HIP_Z = float(T_WORLD_BASE[2, 3])
FOOT_Y = -0.10
DEFAULT_Q = np.array((0.0, 0.02, 1.17), dtype=float)

PERIOD = 2.0
STEP_LENGTH = 0.10
STEP_HEIGHT = 0.05
STAND_HEIGHT = 0.14

CONFIG_DT = 0.005
MAX_TORQUE = 3.0
PD_GAINS = (
    PDGains(kp=18.0, kd=1.2),
    PDGains(kp=24.0, kd=1.4),
    PDGains(kp=12.0, kd=0.8),
)

GIF_SECONDS = 8.0
GIF_FPS = 15
FRAME_SIZE = (720, 400)
SCENE_TARGET_Y_PX = 215.0
BROKEN_TAIL_SECONDS = 2.0
SWEEP_SECONDS = 12.0


@dataclass
class IkContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    foot_id: int


@dataclass
class IkResult:
    q: np.ndarray
    iters: int
    residual: float
    converged: bool


@dataclass
class SteppingTrace:
    time: np.ndarray
    target_xyz: np.ndarray
    ik_foot_xyz: np.ndarray
    q_target: np.ndarray
    q_actual: np.ndarray
    ik_iters: np.ndarray
    ik_residual: np.ndarray
    base_z: np.ndarray
    frames: list[np.ndarray] | None = None


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 Lab 3 的 mounted 单腿 MJCF。"""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = CONFIG_DT
    data = mujoco.MjData(model)
    return model, data


def _joint_qpos_qvel_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"缺少关节 {name!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=int), np.asarray(qvel_ids, dtype=int)


def _actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"缺少执行器 {name!r}")
        ids.append(int(actuator_id))
    return np.asarray(ids, dtype=int)


def _site_id(model: mujoco.MjModel, name: str = FOOT_SITE_NAME) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"缺少 site {name!r}")
    return int(site_id)


def check_scene_mounted(model: mujoco.MjModel) -> None:
    """任务 A：确认 scene.xml 只有一条 hip-world weld，且三关节 / 三 motor 都还在。"""

    if model.neq != 1:
        raise AssertionError(f"scene.xml 应只有 1 条 equality，实际 model.neq={model.neq}")
    if int(model.eq_type[0]) != int(mujoco.mjtEq.mjEQ_WELD):
        raise AssertionError("唯一的 equality 必须是 weld")

    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    world_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
    body_pair = {int(model.eq_obj1id[0]), int(model.eq_obj2id[0])}
    if body_pair != {hip_id, world_id}:
        raise AssertionError("weld 必须连接 body1='hip' 和 body2='world'")

    _joint_qpos_qvel_ids(model)
    _actuator_ids(model)


def make_ik_context(model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None) -> IkContext:
    """把 `mj_jacSite` 需要的 MuJoCo 句柄收成一个小 context。"""

    if model is None or data is None:
        model, data = load_model()
    qpos_ids, qvel_ids = _joint_qpos_qvel_ids(model)
    return IkContext(
        model=model,
        data=data,
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        foot_id=_site_id(model),
    )


def raibert_foot_traj(
    t: float,
    *,
    period: float = PERIOD,
    step_length: float = STEP_LENGTH,
    step_height: float = STEP_HEIGHT,
    stand_height: float = STAND_HEIGHT,
    mid_swing_lift: float = 1.0,
) -> np.ndarray:
    """任务 B：返回 `(x, y, z)` 世界坐标足端目标。"""

    if step_length < 0.0:
        raise ValueError("step_length 必须非负")
    if step_height < 0.0:
        raise ValueError("step_height 必须非负")
    if stand_height <= 0.0:
        raise ValueError("stand_height 必须为正数")

    phase = stance_swing_phase(t, period=period, duty=0.5)
    z_stance = HIP_Z - stand_height
    half_step = 0.5 * step_length

    # TODO 1: 填 stance / swing 分支。stance 沿 -x 匀速后扫；
    # swing 从后方回到前方，z 用 mid-swing 抛物线抬高。
    raise NotImplementedError("TODO 1: 写 Raibert stance/swing 轨迹")


def _raibert_without_mid_swing(
    t: float,
    *,
    period: float = PERIOD,
    step_length: float = STEP_LENGTH,
    stand_height: float = STAND_HEIGHT,
) -> np.ndarray:
    """GIF 末尾反例：swing 只走直线，不抬 mid-swing。"""

    phase = stance_swing_phase(t, period=period, duty=0.5)
    z_stance = HIP_Z - stand_height
    half_step = 0.5 * step_length
    if phase.in_stance:
        x = half_step - step_length * phase.local_phase
    else:
        x = -half_step + step_length * phase.local_phase
    return np.array((x, FOOT_Y, z_stance), dtype=float)


def dls_ik_step(
    qpos: np.ndarray,
    target_xyz: np.ndarray,
    *,
    ctx: IkContext,
    lam: float = 0.05,
    step: float = 0.5,
) -> tuple[np.ndarray, float]:
    """任务 C：做一步 DLS IK，返回 `(q_next, residual)`。"""

    qpos = np.asarray(qpos, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if qpos.shape != (3,) or target_xyz.shape != (3,):
        raise ValueError("qpos 和 target_xyz 的形状必须分别是 (3,) 和 (3,)")

    # TODO 2: 写 DLS 一步迭代。提示：先写 qpos，`mj_forward`，再用
    # `mj_jacSite` 取 3x3 平移雅可比，最后 `q + step * dtheta`。
    raise NotImplementedError("TODO 2: 写 DLS IK 一步迭代")


def dls_ik(
    q0: np.ndarray,
    target_xyz: np.ndarray,
    *,
    ctx: IkContext | None = None,
    lam: float = 0.05,
    step: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> IkResult:
    """把足端目标点求成三关节角。"""

    if ctx is None:
        ctx = make_ik_context()
    q = np.clip(np.asarray(q0, dtype=float), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    residual = float("inf")
    for k in range(max_iter):
        q, residual = dls_ik_step(q, target_xyz, ctx=ctx, lam=lam, step=step)
        if residual < tol:
            return IkResult(q=q, iters=k + 1, residual=residual, converged=True)
    return IkResult(q=q, iters=max_iter, residual=residual, converged=False)


def apply_pd_control(
    data: mujoco.MjData,
    actuator_ids: np.ndarray,
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    q_target: np.ndarray,
) -> np.ndarray:
    """任务 D：把 IK 解出的关节目标角接到三路 actuator。"""

    q = data.qpos[qpos_ids].copy()
    qdot = data.qvel[qvel_ids].copy()
    torques = np.empty(3, dtype=float)
    for i, gains in enumerate(PD_GAINS):
        torques[i] = gains.kp * (q_target[i] - q[i]) + gains.kd * (0.0 - qdot[i])
    torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

    # TODO 3: 把 `torques` 写到对应 actuator 的 `data.ctrl[...]` 里。
    raise NotImplementedError("TODO 3: 把 PD torque 接到 actuator")


def simulate_stepping(
    *,
    seconds: float = GIF_SECONDS,
    capture_frames: bool = False,
    sweep_step_length: bool = False,
    broken_tail: bool = False,
) -> SteppingTrace:
    """运行闭环踏步。`capture_frames=True` 时顺便录 GIF 帧。"""

    model, data = load_model()
    check_scene_mounted(model)
    qpos_ids, qvel_ids = _joint_qpos_qvel_ids(model)
    actuator_ids = _actuator_ids(model)
    ctx = make_ik_context(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    q_target = DEFAULT_Q.copy()
    data.qpos[qpos_ids] = q_target
    data.qvel[qvel_ids] = 0.0
    mujoco.mj_forward(model, data)

    renderer = None
    if capture_frames:
        renderer = mujoco.Renderer(model, height=FRAME_SIZE[1], width=FRAME_SIZE[0])

    ts: list[float] = []
    targets: list[np.ndarray] = []
    ik_foots: list[np.ndarray] = []
    q_targets: list[np.ndarray] = []
    q_actuals: list[np.ndarray] = []
    ik_iters: list[int] = []
    ik_residuals: list[float] = []
    base_zs: list[float] = []
    frames: list[np.ndarray] = []
    trail: list[np.ndarray] = []

    dt = float(model.opt.timestep)
    steps = int(round(seconds / dt))
    frame_period = 1.0 / GIF_FPS
    next_frame_time = 0.0
    max_frames = int(round(seconds * GIF_FPS))

    try:
        for step_idx in range(steps):
            t = step_idx * dt
            if sweep_step_length:
                step_length = 0.12 * min(t / seconds, 1.0)
            else:
                step_length = STEP_LENGTH

            in_broken_tail = broken_tail and t >= seconds - BROKEN_TAIL_SECONDS
            if in_broken_tail:
                target = _raibert_without_mid_swing(t, step_length=step_length)
            else:
                target = raibert_foot_traj(t, step_length=step_length)

            ik = dls_ik(q_target, target, ctx=ctx, lam=0.05, step=0.5, tol=1e-3, max_iter=50)
            q_target = ik.q
            torques = apply_pd_control(data, actuator_ids, qpos_ids, qvel_ids, q_target)
            mujoco.mj_step(model, data)

            ik_foot = fk_leg(q_target)
            ts.append(t)
            targets.append(target)
            ik_foots.append(ik_foot)
            q_targets.append(q_target.copy())
            q_actuals.append(data.qpos[qpos_ids].copy())
            ik_iters.append(ik.iters)
            ik_residuals.append(ik.residual)
            base_zs.append(float(data.xpos[base_id, 2]))

            if (
                renderer is not None
                and len(frames) < max_frames
                and t + 0.5 * dt >= next_frame_time
            ):
                trail.append(target.copy())
                renderer.update_scene(data)
                for point in trail[-180:]:
                    _add_marker(renderer.scene, point, rgba=(0.1, 0.45, 0.9, 0.28), radius=0.003)
                _add_marker(renderer.scene, target, rgba=(0.1, 0.9, 0.2, 1.0), radius=0.012)
                _add_marker(renderer.scene, ik_foot, rgba=(0.9, 0.15, 0.12, 1.0), radius=0.012)
                frame = renderer.render()
                frames.append(
                    _caption_frame(
                        frame,
                        t=t,
                        step_length=step_length,
                        step_height=0.0 if in_broken_tail else STEP_HEIGHT,
                        stand_height=STAND_HEIGHT,
                        is_broken=in_broken_tail,
                    )
                )
                next_frame_time += frame_period
    finally:
        if renderer is not None:
            renderer.close()

    return SteppingTrace(
        time=np.asarray(ts),
        target_xyz=np.asarray(targets),
        ik_foot_xyz=np.asarray(ik_foots),
        q_target=np.asarray(q_targets),
        q_actual=np.asarray(q_actuals),
        ik_iters=np.asarray(ik_iters),
        ik_residual=np.asarray(ik_residuals),
        base_z=np.asarray(base_zs),
        frames=frames if capture_frames else None,
    )


def launch_viewer() -> None:
    """打开 passive viewer，看绿色目标点和红色 IK 足端预测。"""

    import mujoco.viewer

    model, data = load_model()
    check_scene_mounted(model)
    qpos_ids, qvel_ids = _joint_qpos_qvel_ids(model)
    actuator_ids = _actuator_ids(model)
    ctx = make_ik_context(model, data)
    q_target = DEFAULT_Q.copy()
    data.qpos[qpos_ids] = q_target

    print("Lab 3 viewer：绿球=Raibert 目标，红球=IK 解出的 foot 预测。关闭 viewer 退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            target = raibert_foot_traj(data.time)
            ik = dls_ik(q_target, target, ctx=ctx)
            q_target = ik.q
            apply_pd_control(data, actuator_ids, qpos_ids, qvel_ids, q_target)
            mujoco.mj_step(model, data)

            viewer.user_scn.ngeom = 0
            _add_marker(viewer.user_scn, target, rgba=(0.1, 0.9, 0.2, 1.0), radius=0.012)
            _add_marker(viewer.user_scn, fk_leg(q_target), rgba=(0.9, 0.15, 0.12, 1.0), radius=0.012)
            viewer.sync()
            time.sleep(float(model.opt.timestep))


def save_raibert_vs_triangle_plot(path: Path) -> None:
    """保存教程抽象三角形 vs Raibert 三角的对比图。

    左子图复刻教程 §3.8 平面臂在 (x, y) 工作空间里的三个顶点；右子图是
    Pupper 单腿在 (x, z) 平面里的 Raibert 三角加 mid-swing 抛物线。两幅图
    坐标系不同——这正是要展示的"抽象几何顶点"和"物理踏步"的距离。
    """

    plot_utils.apply_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    tutorial_vertices = np.array(
        [(0.45, 0.10), (0.30, -0.10), (0.55, -0.05)],
        dtype=float,
    )
    tri = np.vstack([tutorial_vertices, tutorial_vertices[0]])
    axes[0].plot(tri[:, 0], tri[:, 1], "o-", color="tab:gray", linewidth=1.6)
    for (x, y), label in zip(tutorial_vertices, ("v0", "v1", "v2")):
        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[0].set_title("教程 §3.8 抽象三角形 (平面 3-DoF 臂, x-y)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")

    ts = np.linspace(0.0, PERIOD, 200)
    pts = np.asarray([raibert_foot_traj(t) for t in ts])
    axes[1].plot(pts[:, 0], pts[:, 2], color="tab:blue", linewidth=2.0)
    axes[1].scatter(
        [0.5 * STEP_LENGTH, 0.0, -0.5 * STEP_LENGTH],
        [HIP_Z - STAND_HEIGHT] * 3,
        color="tab:orange",
        s=24,
        zorder=3,
        label="touchdown / standing / liftoff",
    )
    axes[1].set_title("Raibert 三角 + mid-swing (Pupper 单腿, x-z)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _caption_frame(
    frame: np.ndarray,
    *,
    t: float,
    step_length: float,
    step_height: float,
    stand_height: float,
    is_broken: bool,
) -> np.ndarray:
    panel_x = FRAME_SIZE[0] - 230
    scene = gif_utils.fit_scene_frame(
        frame,
        output_size=FRAME_SIZE,
        content_target=(0.5 * panel_x, SCENE_TARGET_Y_PX),
    )
    image = Image.fromarray(scene)
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(18)
    fill = (255, 245, 245, 232) if is_broken else (255, 255, 255, 222)
    text_color = (180, 0, 0, 255) if is_broken else (20, 20, 20, 255)
    draw.rectangle((panel_x, 0, image.width, image.height), fill=fill)

    if is_broken:
        lines = (
            "反例：去掉",
            "mid-swing",
            "",
            "swing 变直线",
            "腿像方波切换",
            "",
            f"t = {t:4.2f} s",
        )
    else:
        lines = (
            "Raibert 悬空踏步",
            "",
            f"step_length = {step_length:.3f} m",
            f"step_height = {step_height:.3f} m",
            f"stand_height = {stand_height:.3f} m",
            f"t = {t:4.2f} s",
            "",
            "绿=目标  红=IK",
        )
    y = 18
    for line in lines:
        draw.text((panel_x + 16, y), line, fill=text_color, font=font)
        y += 30
    return np.asarray(image)


def main() -> None:
    model, _ = load_model()
    check_scene_mounted(model)
    q_truth = np.array((0.0, -0.9, 1.3), dtype=float)
    target = fk_leg(q_truth)
    ik = dls_ik(DEFAULT_Q, target, tol=1e-3, max_iter=50)
    print("CS123 Lab 3：Pupper 单腿 Raibert 悬空踏步")
    print(f"scene weld 数量：model.neq={model.neq}")
    print(f"IK 自检 residual = {ik.residual:.2e} m, iters = {ik.iters}, converged = {ik.converged}")


if __name__ == "__main__":
    if "--viewer" in sys.argv:
        launch_viewer()
    else:
        main()
