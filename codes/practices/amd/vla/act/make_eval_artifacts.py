"""把 LeRobot 评测结果整理成教程可直接引用的图像和视频。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_INFO = (
    LAB_DIR / "outputs" / "eval_act_aloha_transfer_rocm_10k" / "eval_info.json"
)
DEFAULT_REPORT_DIR = LAB_DIR / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ACT 评测教程素材")
    parser.add_argument("--eval-info", type=Path, default=DEFAULT_EVAL_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--gif-fps", type=float, default=12.5)
    parser.add_argument("--gif-width", type=int, default=480)
    args = parser.parse_args()
    if args.gif_fps <= 0:
        parser.error("--gif-fps 必须大于 0")
    if args.gif_width < 160:
        parser.error("--gif-width 不能小于 160")
    return args


def resolve_video_path(raw_path: str, eval_info_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [path, LAB_DIR / path, eval_info_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到评测视频：{raw_path}")


def video_metadata(path: Path) -> tuple[int, float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 50.0)
        frame_count = stream.frames
        if not frame_count:
            frame_count = sum(1 for _ in container.decode(video=0))
    return frame_count, fps


def read_video(path: Path) -> tuple[list[Image.Image], float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 50.0)
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"视频没有可解码帧：{path}")
    return frames, fps


def save_gif(
    frames: list[Image.Image],
    source_fps: float,
    output_path: Path,
    target_fps: float,
    width: int,
) -> None:
    stride = max(round(source_fps / target_fps), 1)
    sampled = frames[::stride]
    height = round(sampled[0].height * width / sampled[0].width)
    resized = [
        frame.resize((width, height), Image.Resampling.LANCZOS).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=192
        )
        for frame in sampled
    ]
    duration_ms = round(1000.0 * stride / source_fps)
    resized[0].save(
        output_path,
        save_all=True,
        append_images=resized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def save_filmstrip(
    frames: list[Image.Image],
    fps: float,
    episode_index: int,
    output_path: Path,
) -> None:
    panel_width, panel_height = 320, 240
    label_height = 42
    indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
    canvas = Image.new("RGB", (panel_width * 5, panel_height + label_height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    for panel, frame_index in enumerate(indices):
        frame = frames[frame_index].resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        x = panel * panel_width
        canvas.paste(frame, (x, 0))
        progress = round(100 * frame_index / max(len(frames) - 1, 1))
        label = f"{progress:>3}%  ·  t={frame_index / fps:.1f}s"
        draw.text((x + 12, panel_height + 10), label, fill="#0f172a", font=font)
        if panel:
            draw.line((x, 0, x, panel_height), fill="#ffffff", width=3)
    canvas.save(output_path, quality=92)


def save_evaluation_plot(
    sum_rewards: list[float],
    max_rewards: list[float],
    successes: list[bool],
    output_dir: Path,
    steps: int,
) -> None:
    episodes = np.arange(len(successes))
    success_count = sum(successes)
    success_rate = 100.0 * success_count / len(successes)
    colors = ["#059669" if success else "#93c5fd" for success in successes]

    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#ffffff")
    axis.bar(episodes, sum_rewards, color=colors, width=0.74)
    axis.set_xticks(episodes, [str(index) for index in episodes])
    axis.set_xlabel("evaluation episode")
    axis.set_ylabel("sum reward")
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.6)
    axis.spines[["top", "right"]].set_visible(False)

    reward_axis = axis.twinx()
    reward_axis.plot(
        episodes,
        max_rewards,
        color="#d97706",
        marker="o",
        markersize=5,
        linewidth=1.4,
        label="max reward",
    )
    reward_axis.set_ylim(-0.15, 4.35)
    reward_axis.set_yticks(range(5))
    reward_axis.set_ylabel("max reward")
    reward_axis.spines["top"].set_visible(False)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#059669", label="success"),
        Patch(facecolor="#93c5fd", label="not completed"),
        reward_axis.lines[0],
    ]
    axis.legend(handles=handles, loc="upper left", frameon=False, ncols=3)
    figure.suptitle(
        "ACT evaluation · ALOHA Transfer Cube\n"
        f"{steps:,} optimizer steps · {success_count}/{len(successes)} successful "
        f"({success_rate:.0f}%)",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    figure.savefig(output_dir / "evaluation_results.webp", dpi=160)
    plt.close(figure)


def build_artifacts(eval_info_path: Path, output_dir: Path, gif_fps: float, gif_width: int) -> dict:
    eval_info_path = eval_info_path.resolve()
    if not eval_info_path.is_file():
        raise FileNotFoundError(f"评测结果不存在：{eval_info_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_info = json.loads(eval_info_path.read_text(encoding="utf-8"))
    task_metrics = eval_info["per_task"][0]["metrics"]
    overall = eval_info["overall"]
    sum_rewards = task_metrics["sum_rewards"]
    max_rewards = task_metrics["max_rewards"]
    successes = task_metrics["successes"]
    raw_video_paths = task_metrics["video_paths"]
    video_paths = [resolve_video_path(path, eval_info_path) for path in raw_video_paths]

    candidates = []
    for episode_index, video_path in enumerate(video_paths):
        if successes[episode_index]:
            frame_count, fps = video_metadata(video_path)
            candidates.append((frame_count / fps, episode_index, video_path))
    if not candidates:
        raise RuntimeError("已录制的前几个回合中没有成功样例，无法生成成功演示视频。")
    duration_s, selected_episode, source_video = min(candidates)

    demo_video = output_dir / "act_aloha_transfer_demo.mp4"
    demo_gif = output_dir / "act_aloha_transfer_demo.gif"
    filmstrip = output_dir / "act_aloha_transfer_keyframes.webp"
    shutil.copy2(source_video, demo_video)
    frames, fps = read_video(source_video)
    save_gif(frames, fps, demo_gif, gif_fps, gif_width)
    save_filmstrip(frames, fps, selected_episode, filmstrip)
    save_evaluation_plot(sum_rewards, max_rewards, successes, output_dir, steps=10001)
    shutil.copy2(eval_info_path, output_dir / "eval_info.json")

    success_count = sum(successes)
    summary = {
        "optimizer_steps": 10001,
        "evaluation_episodes": len(successes),
        "successful_episodes": success_count,
        "success_rate_percent": 100.0 * success_count / len(successes),
        "average_sum_reward": overall["avg_sum_reward"],
        "average_max_reward": overall["avg_max_reward"],
        "evaluation_seconds": overall["eval_s"],
        "selected_demo_episode": selected_episode,
        "selected_demo_duration_seconds": duration_s,
        "selected_demo_sum_reward": sum_rewards[selected_episode],
        "selected_demo_max_reward": max_rewards[selected_episode],
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"演示视频：{demo_video}")
    print(f"演示 GIF：{demo_gif}")
    print(f"关键帧：{filmstrip}")
    print(f"评测图：{output_dir / 'evaluation_results.webp'}")
    return summary


def main() -> None:
    args = parse_args()
    build_artifacts(args.eval_info, args.output_dir, args.gif_fps, args.gif_width)


if __name__ == "__main__":
    main()
