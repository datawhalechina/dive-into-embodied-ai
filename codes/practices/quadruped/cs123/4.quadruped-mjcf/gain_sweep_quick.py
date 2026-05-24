"""4.6 小节用的 gainprm / biasprm 参数扫描脚本。

在本目录运行：
    python gain_sweep_quick.py

macOS 上如果要打开 viewer 单独看一组参数，用：
    mjpython gain_sweep_quick.py --viewer default

默认批量模式不会修改 XML 文件。脚本会为每一组参数重新加载
``models/pupper_v3_floating.xml``，只在内存里改 MuJoCo 编译后的 actuator
参数，然后输出：

* outputs/gain_sweep_summary.csv
* outputs/gain_sweep_base_z.png
* outputs/gain_sweep_effects.png
* outputs/gain_sweep_effects.gif

默认会在 t=2s 给 base 一个短暂向下外力，方便在曲线和 GIF 里看出 Kp / Kd
差异。只想看自由落体后站稳的过程，可以加 ``--no-push``。
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(pathlib.Path(tempfile.gettempdir()) / "cs123-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw


HERE = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "pupper_v3_floating.xml"
DEFAULT_OUT_DIR = HERE / "outputs"

FRAME_SIZE = (360, 270)
GIF_FPS = 8
SIM_SECONDS = 10.0
PUSH_START = 2.0
PUSH_DURATION = 0.25
PUSH_FORCE_Z = -25.0


@dataclass(frozen=True)
class GainCase:
    key: str
    kp: float
    kd: float
    note: str

    @property
    def gainprm(self) -> str:
        return f"{self.kp:g} 0 0"

    @property
    def biasprm(self) -> str:
        return f"0 {-self.kp:g} {-self.kd:g}"


@dataclass
class CaseResult:
    case: GainCase
    times: np.ndarray
    heights: np.ndarray
    final_z: float
    last_second_std_mm: float
    pre_push_z: float
    push_dip_mm: float
    frames: list[Image.Image]
    push_start: float | None
    push_duration: float
    push_force_z: float


CASES: tuple[GainCase, ...] = (
    GainCase("soft", 1.0, 0.05, "Kp 太小，腿会被自重压低"),
    GainCase("default", 5.0, 0.1, "仓库默认值，能稳定站住"),
    GainCase("stiff", 30.0, 0.1, "Kp 偏大，抗扰动强但接触更硬"),
    GainCase("underD", 5.0, 0.0, "Kd 为 0，缺少速度阻尼"),
)


def apply_pd(model: mujoco.MjModel, kp: float, kd: float) -> None:
    """把 XML 里的 gainprm / biasprm 等价地写进编译后的 model."""

    # 对应 XML:
    #   gainprm = "Kp 0 0"
    #   biasprm = "0 -Kp -Kd"
    # 这里先清零其余列，避免学习者后面改过 XML 后留下额外项。
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd


def load_case(case: GainCase) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """加载一份干净模型，应用一组 PD 参数，并回到 home keyframe."""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    apply_pd(model, case.kp, case.kd)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    # home keyframe 里 ctrl 是 12 个目标关节角。每步重新写入，避免被别的逻辑改掉。
    target_ctrl = data.ctrl.copy()
    mujoco.mj_forward(model, data)
    return model, data, target_ctrl


def base_body_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if body_id < 0:
        raise ValueError('body "base_link" not found')
    return body_id


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.14]
    camera.distance = 0.85
    camera.azimuth = 135
    camera.elevation = -22
    return camera


def caption_frame(frame: np.ndarray, case: GainCase, t: float) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 54), fill=(255, 255, 255))
    draw.text((10, 8), f"{case.key}: Kp={case.kp:g}, Kd={case.kd:g}", fill=(20, 20, 20))
    draw.text(
        (10, 29),
        f"gainprm={case.gainprm}   biasprm={case.biasprm}   t={t:4.1f}s",
        fill=(55, 55, 55),
    )
    return image


def try_make_renderer(model: mujoco.MjModel) -> mujoco.Renderer | None:
    try:
        return mujoco.Renderer(model, height=FRAME_SIZE[1], width=FRAME_SIZE[0])
    except Exception as exc:  # pragma: no cover - depends on local OpenGL setup.
        print(
            "当前环境不能使用 MuJoCo offscreen renderer；"
            "脚本仍会生成数值 CSV 和 base z 曲线。",
            file=sys.stderr,
        )
        print(f"Renderer error: {exc}", file=sys.stderr)
        return None


def simulate_case(
    case: GainCase,
    *,
    seconds: float,
    fps: int,
    render: bool,
    push_start: float | None,
    push_duration: float,
    push_force_z: float,
) -> CaseResult:
    model, data, target_ctrl = load_case(case)
    base_id = base_body_id(model)
    camera = make_camera()
    renderer = try_make_renderer(model) if render else None

    times: list[float] = []
    heights: list[float] = []
    frames: list[Image.Image] = []
    next_frame_time = 0.0
    frame_interval = 1.0 / max(fps, 1)

    try:
        if renderer is not None:
            renderer.update_scene(data, camera=camera)
            frames.append(caption_frame(renderer.render(), case, data.time))
            next_frame_time = frame_interval

        while data.time < seconds:
            data.ctrl[:] = target_ctrl
            data.xfrc_applied[:, :] = 0.0
            if (
                push_start is not None
                and push_start <= data.time < push_start + push_duration
            ):
                data.xfrc_applied[base_id, 2] = push_force_z
            mujoco.mj_step(model, data)
            times.append(float(data.time))
            heights.append(float(data.qpos[2]))

            if renderer is not None and data.time + 1e-12 >= next_frame_time:
                renderer.update_scene(data, camera=camera)
                frames.append(caption_frame(renderer.render(), case, data.time))
                next_frame_time += frame_interval
    finally:
        if renderer is not None:
            renderer.close()

    height_array = np.asarray(heights, dtype=np.float64)
    time_array = np.asarray(times, dtype=np.float64)
    samples_per_second = max(int(round(1.0 / model.opt.timestep)), 1)
    tail = height_array[-min(samples_per_second, len(height_array)) :]
    pre_push_z, push_dip_mm = push_response_metrics(
        time_array,
        height_array,
        push_start=push_start,
        push_duration=push_duration,
    )
    return CaseResult(
        case=case,
        times=time_array,
        heights=height_array,
        final_z=float(tail[-1]) if len(tail) else float("nan"),
        last_second_std_mm=float(np.std(tail) * 1000.0) if len(tail) else float("nan"),
        pre_push_z=pre_push_z,
        push_dip_mm=push_dip_mm,
        frames=frames,
        push_start=push_start,
        push_duration=push_duration,
        push_force_z=push_force_z,
    )


def push_response_metrics(
    times: np.ndarray,
    heights: np.ndarray,
    *,
    push_start: float | None,
    push_duration: float,
) -> tuple[float, float]:
    if push_start is None or len(times) == 0:
        return float("nan"), float("nan")

    before = (times >= max(0.0, push_start - 0.5)) & (times < push_start)
    after = (times >= push_start) & (times < push_start + max(push_duration, 0.0) + 1.5)
    if not np.any(before) or not np.any(after):
        return float("nan"), float("nan")

    pre_push_z = float(np.mean(heights[before]))
    min_after_push_z = float(np.min(heights[after]))
    return pre_push_z, max((pre_push_z - min_after_push_z) * 1000.0, 0.0)


def print_summary(results: list[CaseResult]) -> None:
    print(
        "case      Kp     Kd     final_z(m)   last_1s_std(mm)   "
        "push_dip(mm)   gainprm        biasprm"
    )
    print("-" * 106)
    for result in results:
        case = result.case
        print(
            f"{case.key:7s} "
            f"{case.kp:6.2f} "
            f"{case.kd:6.2f} "
            f"{result.final_z:11.3f} "
            f"{result.last_second_std_mm:16.2f}   "
            f"{result.push_dip_mm:12.2f}   "
            f"{case.gainprm:12s} {case.biasprm}"
        )


def save_summary_csv(results: list[CaseResult], out_dir: pathlib.Path) -> pathlib.Path:
    path = out_dir / "gain_sweep_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "kp",
                "kd",
                "gainprm",
                "biasprm",
                "final_z_m",
                "last_1s_std_mm",
                "pre_push_z_m",
                "push_dip_mm",
                "note",
            ]
        )
        for result in results:
            case = result.case
            writer.writerow(
                [
                    case.key,
                    case.kp,
                    case.kd,
                    case.gainprm,
                    case.biasprm,
                    f"{result.final_z:.6f}",
                    f"{result.last_second_std_mm:.6f}",
                    f"{result.pre_push_z:.6f}",
                    f"{result.push_dip_mm:.6f}",
                    case.note,
                ]
            )
    return path


def save_height_plot(results: list[CaseResult], out_dir: pathlib.Path) -> pathlib.Path:
    path = out_dir / "gain_sweep_base_z.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    for result in results:
        label = (
            f"{result.case.key} "
            f"(Kp={result.case.kp:g}, Kd={result.case.kd:g}, "
            f"dip={result.push_dip_mm:.1f} mm)"
        )
        ax.plot(result.times, result.heights, linewidth=1.5, label=label)
    first = results[0] if results else None
    if first is not None and first.push_start is not None:
        ax.axvspan(
            first.push_start,
            first.push_start + first.push_duration,
            color="#d55e00",
            alpha=0.16,
            label=f"downward push ({first.push_force_z:g} N)",
        )
    ax.set_title("Floating Pupper base height under different PD gains")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("base z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_effect_panel(results: list[CaseResult], out_dir: pathlib.Path) -> pathlib.Path | None:
    last_frames = [result.frames[-1] for result in results if result.frames]
    if len(last_frames) != len(results):
        return None

    panel_w, panel_h = FRAME_SIZE
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), (245, 245, 245))
    for idx, frame in enumerate(last_frames):
        resized = frame.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = (idx % 2) * panel_w
        y = (idx // 2) * panel_h
        canvas.paste(resized, (x, y))

    path = out_dir / "gain_sweep_effects.png"
    canvas.save(path)
    return path


def save_effect_gif(
    results: list[CaseResult],
    out_dir: pathlib.Path,
    *,
    fps: int,
) -> pathlib.Path | None:
    if any(not result.frames for result in results):
        return None

    frame_count = min(len(result.frames) for result in results)
    panel_w, panel_h = FRAME_SIZE
    combined_frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), (245, 245, 245))
        for case_idx, result in enumerate(results):
            frame = result.frames[frame_idx].resize((panel_w, panel_h), Image.Resampling.LANCZOS)
            x = (case_idx % 2) * panel_w
            y = (case_idx // 2) * panel_h
            canvas.paste(frame, (x, y))
        combined_frames.append(canvas)

    path = out_dir / "gain_sweep_effects.gif"
    combined_frames[0].save(
        path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=int(round(1000 / max(fps, 1))),
        loop=0,
    )
    return path


def run_batch(args: argparse.Namespace) -> int:
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        simulate_case(
            case,
            seconds=args.seconds,
            fps=args.fps,
            render=not args.no_render,
            push_start=None if args.no_push else args.push_start,
            push_duration=args.push_duration,
            push_force_z=args.push_force_z,
        )
        for case in CASES
    ]
    print_summary(results)

    csv_path = save_summary_csv(results, out_dir)
    plot_path = save_height_plot(results, out_dir)
    panel_path = save_effect_panel(results, out_dir)
    gif_path = save_effect_gif(results, out_dir, fps=args.fps)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {plot_path}")
    if panel_path is not None:
        print(f"  {panel_path}")
    if gif_path is not None:
        print(f"  {gif_path}")
    if panel_path is None or gif_path is None:
        print("  visual render skipped; run without --no-render on a machine with MuJoCo OpenGL.")

    return 0


def run_viewer(args: argparse.Namespace) -> int:
    case_by_key = {case.key: case for case in CASES}
    case = case_by_key[args.viewer]
    model, data, target_ctrl = load_case(case)
    base_id = base_body_id(model)
    push_start = None if args.no_push else args.push_start

    print(
        f"Opening {case.key}: Kp={case.kp:g}, Kd={case.kd:g}, "
        f"gainprm={case.gainprm}, biasprm={case.biasprm}",
        flush=True,
    )
    if push_start is not None:
        print(
            f"Applying a short downward push at t={push_start:g}s "
            f"({args.push_force_z:g} N for {args.push_duration:g}s).",
            flush=True,
        )
    print("Close the viewer window to print final z / last-1s std.", flush=True)

    import mujoco.viewer

    heights: list[float] = []
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.14]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -22
            while viewer.is_running():
                step_start = time.perf_counter()
                data.ctrl[:] = target_ctrl
                data.xfrc_applied[:, :] = 0.0
                if (
                    push_start is not None
                    and push_start <= data.time < push_start + args.push_duration
                ):
                    data.xfrc_applied[base_id, 2] = args.push_force_z
                mujoco.mj_step(model, data)
                heights.append(float(data.qpos[2]))
                viewer.sync()

                elapsed = time.perf_counter() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            print("\nOn macOS, run the interactive viewer with mjpython:", file=sys.stderr)
            print(f"  mjpython {pathlib.Path(__file__)} --viewer {case.key}", file=sys.stderr)
            return 2
        raise

    if heights:
        samples_per_second = max(int(round(1.0 / model.opt.timestep)), 1)
        tail = np.asarray(heights[-min(samples_per_second, len(heights)) :], dtype=np.float64)
        print(f"final z={tail[-1]:.3f} m, last-1s std={np.std(tail) * 1000.0:.2f} mm")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 4.6 小节的 gainprm / biasprm 参数组合，不修改 XML。",
    )
    parser.add_argument("--seconds", type=float, default=SIM_SECONDS, help="仿真时长")
    parser.add_argument("--fps", type=int, default=GIF_FPS, help="GIF 渲染帧率")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--no-render", action="store_true", help="跳过 PNG/GIF 渲染")
    parser.add_argument("--no-push", action="store_true", help="关闭 t=2s 的下压扰动")
    parser.add_argument("--push-start", type=float, default=PUSH_START, help="扰动开始时间")
    parser.add_argument("--push-duration", type=float, default=PUSH_DURATION, help="扰动持续时间")
    parser.add_argument("--push-force-z", type=float, default=PUSH_FORCE_Z, help="世界系 z 方向外力")
    parser.add_argument(
        "--viewer",
        choices=[case.key for case in CASES],
        help="打开某一组参数的交互 viewer，不跑批量模式",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.viewer:
        return run_viewer(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
