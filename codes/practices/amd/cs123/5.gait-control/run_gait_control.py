"""在真实 Pupper v3 上运行 §5 的开环步态实验（本目录自包含）。

管线：`pupper_ik` 的数值 IK 把足端轨迹反解成关节角 → 写进真实模型自带的位置
伺服 → MjSpec 加的 base weld 把机身固定住 → 离屏渲染成 GIF。支持 walk、trot、
pace、bound 和 gallop，原地与前进实验只差步长和是否平移焊接点。可在 MuJoCo viewer 中交互
预览，也可离屏渲染 GIF 到本目录 `outputs/`。

在 cs123 目录下运行：
    uv run python 5.gait-control/run_gait_control.py
    uv run mjpython 5.gait-control/run_gait_control.py --viewer inplace  # macOS
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupper_ik import (  # noqa: E402
    HIP_OFFSETS,
    LEG_ORDER,
    WELD_RELPOSE_X,
    PupperLegIK,
    build_sim_model,
)


OUT_DIR = Path(__file__).with_name("outputs")

WIDTH = 720
HEIGHT = 540
FPS = 24
SETTLE_SECONDS = 0.35
RENDER_SECONDS = 3.0
COMPARISON_GAITS = ("walk", "pace", "bound", "gallop")
COMPARISON_PANEL_SIZE = (400, 300)
COMPARISON_GAP = 4


@dataclass(frozen=True)
class Gait:
    key: str
    phase_offsets: dict[str, float]
    duty: float
    inplace_t_cycle: float
    forward_t_cycle: float
    inplace_step_height: float
    forward_step_height: float
    forward_step_length: float
    weld_speed: float


GAITS = {
    gait.key: gait
    for gait in (
        # 慢步：四条腿依次落足，大部分时间保持三足支撑。
        Gait(
            key="walk",
            phase_offsets={"RL": 0.00, "FL": 0.25, "RR": 0.50, "FR": 0.75},
            duty=0.75,
            inplace_t_cycle=0.9,
            forward_t_cycle=0.9,
            inplace_step_height=0.025,
            forward_step_height=0.030,
            forward_step_length=0.035,
            weld_speed=0.04,
        ),
        # 小跑：对角腿同相，两组错半个周期。
        Gait(
            key="trot",
            phase_offsets={"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0},
            duty=0.50,
            inplace_t_cycle=0.4,
            forward_t_cycle=0.5,
            inplace_step_height=0.030,
            forward_step_height=0.035,
            forward_step_length=0.050,
            weld_speed=0.10,
        ),
        # 侧对步：同侧前后腿同相，左右两组错半个周期。
        Gait(
            key="pace",
            phase_offsets={"FL": 0.0, "FR": 0.5, "RL": 0.0, "RR": 0.5},
            duty=0.50,
            inplace_t_cycle=0.4,
            forward_t_cycle=0.5,
            inplace_step_height=0.030,
            forward_step_height=0.035,
            forward_step_length=0.050,
            weld_speed=0.10,
        ),
        # 跳跃：两条后腿同相，两条前腿同相，前后错半周期。
        Gait(
            key="bound",
            phase_offsets={"RL": 0.0, "RR": 0.0, "FL": 0.5, "FR": 0.5},
            duty=0.42,
            inplace_t_cycle=0.45,
            forward_t_cycle=0.45,
            inplace_step_height=0.040,
            forward_step_height=0.045,
            forward_step_length=0.065,
            weld_speed=0.14,
        ),
        # 奔驰：旋转式四拍落足 RR → RL → FL → FR，周期末保留腾空相。
        Gait(
            key="gallop",
            phase_offsets={"RR": 0.00, "RL": 0.12, "FL": 0.45, "FR": 0.57},
            duty=0.28,
            inplace_t_cycle=0.55,
            forward_t_cycle=0.50,
            inplace_step_height=0.045,
            forward_step_height=0.050,
            forward_step_length=0.080,
            weld_speed=0.18,
        ),
    )
}


@dataclass(frozen=True)
class Experiment:
    gait: Gait
    key: str
    name: str
    output: Path
    step_length: float
    t_cycle: float
    step_height: float
    stand_height: float
    weld_speed: float = 0.0


def make_experiments(gait: Gait) -> dict[str, Experiment]:
    return {
        experiment.key: experiment
        for experiment in (
            Experiment(
                gait=gait,
                key="inplace",
                name=f"In-place {gait.key} · pupper_v3.xml",
                output=OUT_DIR / f"lab5_inplace_{gait.key}.gif",
                step_length=0.0,
                t_cycle=gait.inplace_t_cycle,
                step_height=gait.inplace_step_height,
                stand_height=0.13,
            ),
            Experiment(
                gait=gait,
                key="forward",
                name=f"Forward {gait.key} · pupper_v3.xml",
                output=OUT_DIR / f"lab5_forward_{gait.key}.gif",
                step_length=gait.forward_step_length,
                t_cycle=gait.forward_t_cycle,
                step_height=gait.forward_step_height,
                stand_height=0.13,
                weld_speed=gait.weld_speed,
            ),
        )
    }


def leg_phase(t: float, leg: str, t_cycle: float, gait: Gait) -> tuple[bool, float]:
    t_local = ((t / t_cycle) - gait.phase_offsets[leg]) % 1.0
    if t_local < gait.duty:
        return True, t_local / gait.duty
    return False, (t_local - gait.duty) / (1.0 - gait.duty)


def foot_trajectory(s: float, in_stance: bool, step_length: float, step_height: float, stand_height: float) -> np.ndarray:
    if in_stance:
        x = step_length * (0.5 - s)
        z = -stand_height
    else:
        x = step_length * (s - 0.5)
        z = -stand_height + step_height * np.sin(np.pi * s)
    return np.array((x, 0.0, z), dtype=float)


def gait_step(
    kin: PupperLegIK,
    t: float,
    step_length: float,
    seed: dict[str, np.ndarray],
    *,
    t_cycle: float,
    step_height: float,
    stand_height: float,
    gait: Gait,
) -> dict[str, np.ndarray]:
    target = {}
    for leg in LEG_ORDER:
        in_stance, s = leg_phase(t, leg, t_cycle, gait)
        foot_base = HIP_OFFSETS[leg] + foot_trajectory(s, in_stance, step_length, step_height, stand_height)
        q = kin.ik(foot_base, leg=leg, q_seed=seed[leg])
        seed[leg] = q
        target[leg] = q
    return target


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.72
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.lookat[:] = (0.0, 0.0, 0.10)
    return camera


def prepare_experiment(
    experiment: Experiment,
) -> tuple[PupperLegIK, mujoco.MjModel, mujoco.MjData, int, dict[str, np.ndarray], dict[str, np.ndarray]]:
    kin = PupperLegIK()
    model = build_sim_model(stand_height=experiment.stand_height, weld=True)
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    seed = {leg: np.zeros(3) for leg in LEG_ORDER}
    q0 = gait_step(
        kin,
        0.0,
        experiment.step_length,
        seed,
        t_cycle=experiment.t_cycle,
        step_height=experiment.step_height,
        stand_height=experiment.stand_height,
        gait=experiment.gait,
    )

    # 初始摆到站姿：base 在 stand_height，12 个关节按 q0。
    data.qpos[:] = 0.0
    data.qpos[2] = experiment.stand_height
    data.qpos[3] = 1.0
    for leg in LEG_ORDER:
        data.qpos[kin.qadr[leg]] = q0[leg]
    mujoco.mj_forward(model, data)
    return kin, model, data, base_id, seed, q0


def step_experiment(
    experiment: Experiment,
    kin: PupperLegIK,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: dict[str, np.ndarray],
    q0: dict[str, np.ndarray],
) -> None:
    gait_t = max(0.0, data.time - SETTLE_SECONDS)
    if experiment.weld_speed and model.neq:
        model.eq_data[0, WELD_RELPOSE_X] = experiment.weld_speed * gait_t
    target = (
        q0
        if data.time < SETTLE_SECONDS
        else gait_step(
            kin,
            gait_t,
            experiment.step_length,
            seed,
            t_cycle=experiment.t_cycle,
            step_height=experiment.step_height,
            stand_height=experiment.stand_height,
            gait=experiment.gait,
        )
    )
    for leg in LEG_ORDER:
        data.ctrl[kin.ctrl[leg]] = target[leg]
    mujoco.mj_step(model, data)


def render_experiment(experiment: Experiment) -> None:
    kin, model, data, base_id, seed, q0 = prepare_experiment(experiment)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = make_camera()
    frames: list[Image.Image] = []
    next_frame_time = 0.0
    base_z: list[float] = []

    try:
        while data.time < SETTLE_SECONDS + RENDER_SECONDS:
            step_experiment(experiment, kin, model, data, seed, q0)

            if data.time < SETTLE_SECONDS:
                continue
            base_z.append(float(data.xpos[base_id][2]))

            render_t = data.time - SETTLE_SECONDS
            if render_t + 0.5 * model.opt.timestep < next_frame_time:
                continue
            camera.lookat[:] = data.xpos[base_id]
            camera.lookat[2] = max(float(camera.lookat[2]), 0.09)
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()).convert("RGB"))
            next_frame_time += 1.0 / FPS
    finally:
        renderer.close()

    experiment.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        experiment.output,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=True,
    )
    std_mm = float(np.std(base_z[-int(round(1.0 / model.opt.timestep)):])) * 1000.0
    print(f"saved {experiment.output} ({len(frames)} frames, base z std={std_mm:.2f} mm)")


def save_gait_comparison(output: Path) -> None:
    """渲染 walk / pace / bound / gallop 的前进实验，并拼成 2×2 GIF。"""
    source_paths = []
    for gait_key in COMPARISON_GAITS:
        experiment = make_experiments(GAITS[gait_key])["forward"]
        render_experiment(experiment)
        source_paths.append(experiment.output)

    panel_width, panel_height = COMPARISON_PANEL_SIZE
    panel_frames: list[list[Image.Image]] = []
    for path in source_paths:
        frames = []
        with Image.open(path) as source:
            # GIF 时间精度为 10 ms，每隔一帧取样得到约 12 FPS。
            for frame_index in range(0, source.n_frames, 2):
                source.seek(frame_index)
                frame = source.convert("RGB").resize(
                    COMPARISON_PANEL_SIZE,
                    Image.Resampling.LANCZOS,
                )
                frames.append(frame)
        panel_frames.append(frames)

    frame_count = min(len(frames) for frames in panel_frames)
    canvas_width = 2 * panel_width + 2 * COMPARISON_GAP
    canvas_height = 2 * panel_height + 2 * COMPARISON_GAP
    positions = (
        (COMPARISON_GAP // 2, COMPARISON_GAP // 2),
        (panel_width + 3 * COMPARISON_GAP // 2, COMPARISON_GAP // 2),
        (COMPARISON_GAP // 2, panel_height + 3 * COMPARISON_GAP // 2),
        (panel_width + 3 * COMPARISON_GAP // 2, panel_height + 3 * COMPARISON_GAP // 2),
    )

    comparison_frames = []
    for frame_index in range(frame_count):
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        for frames, position in zip(panel_frames, positions, strict=True):
            canvas.paste(frames[frame_index], position)
        comparison_frames.append(
            canvas.quantize(
                colors=128,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    comparison_frames[0].save(
        output,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    print(f"saved {output} ({frame_count} frames, 2x2 gait comparison)")


def view_experiment(experiment: Experiment) -> int:
    kin, model, data, base_id, seed, q0 = prepare_experiment(experiment)
    print(f"Opening {experiment.name}. Close the viewer window to exit.", flush=True)

    import mujoco.viewer

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 0.72
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                step_start = time.perf_counter()
                step_experiment(experiment, kin, model, data, seed, q0)
                viewer.cam.lookat[:] = data.xpos[base_id]
                viewer.cam.lookat[2] = max(float(viewer.cam.lookat[2]), 0.09)
                viewer.sync()

                elapsed = time.perf_counter() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            print("\nmacOS 上请用 mjpython 启动交互 viewer：", file=sys.stderr)
            print(
                f"  uv run mjpython {Path(__file__)} "
                f"--gait {experiment.gait.key} --viewer {experiment.key}",
                file=sys.stderr,
            )
            return 2
        raise
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览或渲染 Pupper v3 开环步态实验。")
    parser.add_argument(
        "--gait",
        choices=GAITS,
        default="trot",
        help="步态类型（默认：trot）",
    )
    parser.add_argument(
        "--viewer",
        choices=("inplace", "forward"),
        help="打开原地或前进步态的 MuJoCo 交互 viewer，不带此参数时生成两段 GIF",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="使用 --viewer 时仍先生成两段 GIF",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="生成 walk / pace / bound / gallop 的 2×2 前进步态对比 GIF",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.comparison:
        save_gait_comparison(OUT_DIR / "lab5_forward_gait_comparison.gif")
        return 0
    experiments = make_experiments(GAITS[args.gait])
    if args.gif or args.viewer is None:
        for experiment in experiments.values():
            render_experiment(experiment)
    if args.viewer is not None:
        return view_experiment(experiments[args.viewer])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
