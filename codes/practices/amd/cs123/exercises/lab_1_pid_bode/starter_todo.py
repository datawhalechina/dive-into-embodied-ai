"""Lab 1 TODO starter：Pupper HFE 正弦跟踪和实测 Bode 图。

作者侧先运行：

    uv run python lab_1_pid_bode/starter_todo.py

它应该停在第一个 NotImplementedError。交付学生版时，抽走已填完的
`starter.py`，再把本文件改名为 `starter.py`。
"""

from __future__ import annotations

import math
import sys
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

from shared.controllers.pd_controller import (  # noqa: E402
    BodePoint,
    JointTarget,
    PDGains,
    SimulationConfig,
    TrackingTrace,
    constant_target,
    ensure_parent,
    estimate_sine_response,
    sine_target,
)
from shared.viz import gif_utils, plot_utils  # noqa: E402


LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "models" / "scene.xml"

CONFIG = SimulationConfig(dt=0.005, max_torque=3.0)
AMPLITUDE_RAD = 0.3
DEFAULT_KP = 2.0
DAMPING_RATIO = 0.7
FREQUENCIES_HZ = (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
BODE_SECONDS = 5.0
BODE_SETTLE_SECONDS = 1.0
GIF_SECONDS = 6.0
GIF_FREQUENCY_HZ = 0.3
SINE_SECONDS = 10.0
GIF_FPS = 15
FRAME_SIZE = (640, 400)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = CONFIG.dt
    data = mujoco.MjData(model)
    return model, data


def _joint_qpos_qvel_ids(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"缺少关节 {joint_name!r}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"缺少执行器 {actuator_name!r}")
    return int(actuator_id)


def read_hfe_reflected_inertia(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """任务 B：从 MuJoCo 的稠密质量矩阵里读取 HFE 反射惯量。"""

    mujoco.mj_forward(model, data)
    _, hfe_dof_id = _joint_qpos_qvel_ids(model, CONFIG.joint_name)
    # TODO 1: 用 mujoco.mj_fullM 展开 data.qM，然后返回 HFE 对角项。
    raise NotImplementedError("TODO 1: 用 mujoco.mj_fullM 读取 I_hfe")


def solve_kd(kp: float, inertia: float, zeta: float = DAMPING_RATIO) -> float:
    """任务 B：按二阶系统经验公式选择阻尼。"""

    # TODO 2: 计算 Kd = 2 * zeta * sqrt(Kp * I_hfe)。
    raise NotImplementedError("TODO 2: 根据 Kp、inertia 和 zeta 计算 Kd")


def default_gains() -> tuple[PDGains, float]:
    model, data = load_model()
    inertia = read_hfe_reflected_inertia(model, data)
    kd = solve_kd(DEFAULT_KP, inertia)
    return PDGains(kp=DEFAULT_KP, kd=kd), inertia


def _pd_torque(gains: PDGains, target: JointTarget, q: float, qdot: float) -> float:
    """任务 A/C/D 共用这一条控制律。"""

    # TODO 3: 实现 tau = Kp(q_des - q) + Kd(qdot_des - qdot)。
    raise NotImplementedError("TODO 3: 实现 PD 控制律")


def simulate_tracking(
    *,
    target_fn,
    seconds: float,
    gains: PDGains,
    inertia: float,
    frequency_hz: float | None = None,
    capture_frames: bool = False,
    capture_until_seconds: float = GIF_SECONDS,
    caption: str = "",
) -> TrackingTrace:
    """运行 HFE 控制器并返回日志时间序列。

    渲染是可选项，只给 make_artifacts.py 使用。数值测试会用 capture_frames=False
    调用同一个函数。
    """

    model, data = load_model()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    hfe_qpos_id, hfe_qvel_id = _joint_qpos_qvel_ids(model, CONFIG.joint_name)
    hfe_actuator_id = _actuator_id(model, CONFIG.actuator_name)

    renderer = None
    if capture_frames:
        renderer = mujoco.Renderer(model, height=FRAME_SIZE[1], width=420)

    ts: list[float] = []
    q_des_log: list[float] = []
    qdot_des_log: list[float] = []
    q_log: list[float] = []
    qdot_log: list[float] = []
    torque_log: list[float] = []
    frames: list[np.ndarray] = []
    frame_period = 1.0 / GIF_FPS
    next_frame_time = 0.0
    max_frames = int(round(capture_until_seconds * GIF_FPS))
    steps = int(seconds / CONFIG.dt)

    for step in range(steps):
        t = step * CONFIG.dt
        target = target_fn(t)
        q = float(data.qpos[hfe_qpos_id])
        qdot = float(data.qvel[hfe_qvel_id])
        torque = _pd_torque(gains, target, q, qdot)
        torque = float(np.clip(torque, -CONFIG.max_torque, CONFIG.max_torque))

        data.ctrl[:] = 0.0
        data.ctrl[hfe_actuator_id] = torque
        mujoco.mj_step(model, data)

        ts.append(t)
        q_des_log.append(target.q)
        qdot_des_log.append(target.qdot)
        q_log.append(q)
        qdot_log.append(qdot)
        torque_log.append(torque)

        if (
            renderer is not None
            and len(frames) < max_frames
            and t + 0.5 * CONFIG.dt >= next_frame_time
        ):
            renderer.update_scene(data)
            scene_frame = renderer.render()
            frames.append(
                _compose_frame(
                    scene_frame,
                    np.asarray(ts),
                    np.asarray(q_des_log),
                    np.asarray(q_log),
                    caption=caption,
                )
            )
            next_frame_time += frame_period

    if renderer is not None:
        renderer.close()

    return TrackingTrace(
        time=np.asarray(ts),
        q_des=np.asarray(q_des_log),
        qdot_des=np.asarray(qdot_des_log),
        q=np.asarray(q_log),
        qdot=np.asarray(qdot_log),
        torque=np.asarray(torque_log),
        kp=gains.kp,
        kd=gains.kd,
        inertia=inertia,
        frequency_hz=frequency_hz,
        frames=frames if capture_frames else None,
    )


def run_constant_hold(
    *,
    q_des: float = AMPLITUDE_RAD,
    seconds: float = 5.0,
    gains: PDGains | None = None,
    inertia: float | None = None,
) -> TrackingTrace:
    if gains is None or inertia is None:
        gains, inertia = default_gains()
    return simulate_tracking(
        target_fn=constant_target(q_des),
        seconds=seconds,
        gains=gains,
        inertia=inertia,
    )


def run_sine_tracking(
    *,
    frequency_hz: float = 0.5,
    seconds: float = SINE_SECONDS,
    gains: PDGains | None = None,
    inertia: float | None = None,
    capture_frames: bool = False,
) -> TrackingTrace:
    if gains is None or inertia is None:
        gains, inertia = default_gains()
    caption = f"HFE 正弦跟踪：f={frequency_hz:g} Hz, Kp={gains.kp:g}, Kd={gains.kd:.3g}"
    return simulate_tracking(
        target_fn=sine_target(AMPLITUDE_RAD, frequency_hz),
        seconds=seconds,
        gains=gains,
        inertia=inertia,
        frequency_hz=frequency_hz,
        capture_frames=capture_frames,
        capture_until_seconds=GIF_SECONDS,
        caption=caption,
    )


def run_bode_sweep(
    *,
    gains: PDGains | None = None,
    inertia: float | None = None,
    frequencies_hz: tuple[float, ...] = FREQUENCIES_HZ,
) -> list[BodePoint]:
    if gains is None or inertia is None:
        gains, inertia = default_gains()

    points: list[BodePoint] = []
    for freq in frequencies_hz:
        trace = run_sine_tracking(
            frequency_hz=freq,
            seconds=BODE_SECONDS,
            gains=gains,
            inertia=inertia,
            capture_frames=False,
        )
        points.append(
            estimate_sine_response(
                trace.time,
                trace.q_des,
                trace.q,
                frequency_hz=freq,
                settle_seconds=BODE_SETTLE_SECONDS,
            )
        )
    return points


def run_experiment(*, render: bool = False) -> dict[str, object]:
    """运行任务 A-D，返回 make_artifacts.py 需要的全部数据。"""

    gains, inertia = default_gains()
    hold_trace = run_constant_hold(gains=gains, inertia=inertia)
    sine_gif = run_sine_tracking(
        frequency_hz=GIF_FREQUENCY_HZ,
        seconds=GIF_SECONDS,
        gains=gains,
        inertia=inertia,
        capture_frames=render,
    )
    bode_points = run_bode_sweep(gains=gains, inertia=inertia)
    return {
        "gains": gains,
        "inertia": inertia,
        "hold_trace": hold_trace,
        "sine_gif": sine_gif,
        "bode_points": bode_points,
    }


def save_bode_plot(points: list[BodePoint], path: Path, gains: PDGains, inertia: float) -> None:
    subtitle = f"Kp={gains.kp:g}, Kd={gains.kd:.4g}, I_hfe={inertia:.4g} kg m^2"
    fig = plot_utils.bode_figure(points, subtitle=subtitle)
    ensure_parent(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gif(
    trace: TrackingTrace,
    path: Path,
    *,
    fps: int = GIF_FPS,
    max_frames: int | None = None,
    width: int | None = None,
) -> None:
    if not trace.frames:
        raise ValueError("trace 里没有渲染帧")
    gif_utils.write_gif(trace.frames, path, fps=fps, max_frames=max_frames, width=width)


def _prepare_scene_frame(scene_frame: np.ndarray) -> np.ndarray:
    return gif_utils.fit_scene_frame(scene_frame, content_target=(210.0, 215.0))


def _compose_frame(
    scene_frame: np.ndarray,
    time: np.ndarray,
    q_des: np.ndarray,
    q: np.ndarray,
    *,
    caption: str,
) -> np.ndarray:
    image = Image.new("RGB", FRAME_SIZE, (0, 0, 0))
    image.paste(Image.fromarray(_prepare_scene_frame(scene_frame)), (0, 0))

    panel = Image.new("RGB", (220, FRAME_SIZE[1]), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    title_font = gif_utils.load_font(17)
    chart_title_font = gif_utils.load_font(13)
    label_font = gif_utils.load_font(11)
    caption_head, sep, caption_tail = caption.partition("：")
    draw.text((14, 16), caption_head, fill=(30, 30, 30), font=title_font)
    if sep:
        draw.text((14, 44), caption_tail.strip(), fill=(30, 30, 30), font=label_font)
    draw.text((14, 88), "目标 vs 实际", fill=(30, 30, 30), font=chart_title_font)
    draw.line((30, 215, 198, 215), fill=(180, 180, 180), width=1)
    draw.line((30, 130, 30, 300), fill=(180, 180, 180), width=1)

    if len(time) > 1:
        t0 = max(0.0, float(time[-1]) - 2.0)
        mask = time >= t0
        tt = time[mask]
        target = q_des[mask]
        actual = q[mask]
        _draw_series(draw, tt, target, color=(210, 60, 60), dashed=True)
        _draw_series(draw, tt, actual, color=(40, 95, 190), dashed=False)

    draw.line((40, 365, 70, 365), fill=(210, 60, 60), width=2)
    draw.text((78, 356), "q_des", fill=(80, 80, 80), font=label_font)
    draw.line((40, 384, 70, 384), fill=(40, 95, 190), width=2)
    draw.text((78, 375), "q_hfe", fill=(80, 80, 80), font=label_font)
    image.paste(panel, (420, 0))
    return np.asarray(image)


def _draw_series(
    draw: ImageDraw.ImageDraw,
    time: np.ndarray,
    values: np.ndarray,
    *,
    color: tuple[int, int, int],
    dashed: bool,
) -> None:
    if len(time) < 2:
        return
    x_min = float(time[0])
    x_max = float(time[-1])
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min, y_max = -0.4, 0.4
    points: list[tuple[float, float]] = []
    for t, value in zip(time, values, strict=False):
        x = 30 + 168 * (float(t) - x_min) / (x_max - x_min)
        y = 215 - 85 * (float(value) - 0.0) / (y_max - y_min)
        points.append((x, y))
    if dashed:
        for idx in range(0, len(points) - 1, 4):
            draw.line(points[idx : idx + 2], fill=color, width=2)
    else:
        draw.line(points, fill=color, width=2)


def main() -> None:
    print("CS123 Lab 1：HFE 正弦跟踪")
    gains, inertia = default_gains()
    hold = run_constant_hold(gains=gains, inertia=inertia)
    final_error = abs(float(hold.q[-1] - AMPLITUDE_RAD))
    print(f"I_hfe={inertia:.6g}, Kp={gains.kp:.3g}, Kd={gains.kd:.6g}")
    print(f"任务 A 最终误差：{final_error:.4f} rad")
    print("Stretch 提示：任务 A-D 跑通后，可以比较 Kp*、2Kp*、5Kp* 三条 Bode 曲线。")


if __name__ == "__main__":
    main()
