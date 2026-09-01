"""使用 LeRobotDataset 在 ALOHA Transfer Cube 数据上训练 ACT。

这是教学用的最小训练循环。它展示 LeRobot 中数据特征、动作时间窗、ACT
配置、预处理器和 checkpoint 保存如何连接起来，不替代功能更完整的
``lerobot-train`` 命令。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.common.train_utils import (
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_OUTPUT_DIR = LAB_DIR / "outputs" / "act_aloha_transfer"
METRIC_FIELDS = (
    "step",
    "elapsed_s",
    "loss",
    "l1_loss",
    "kld_loss",
    "grad_norm",
    "learning_rate",
    "steps_per_s",
    "gpu_memory_gb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LeRobot 的 ACT 实现在 ALOHA 仿真示教上训练策略。",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--revision",
        default="v3.0",
        help="Hugging Face 数据集 revision；固定版本便于复现实验。",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="只加载指定 episode，例如 --episodes 0 1；默认加载全部。",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="关闭 ACT 的 CVAE，便于做消融实验。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1000,
        help="每隔多少 step 更新训练曲线；设为 0 可关闭中途绘图。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help="每隔多少 step 保存可恢复 checkpoint；设为 0 只保存最终结果。",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="从 checkpoint 目录恢复，例如 outputs/.../checkpoints/last。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="在 CUDA 上启用 bfloat16 自动混合精度。",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="训练墙钟时限；到点后完成当前 step 并安全保存。",
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.num_workers < 0:
        parser.error("--num-workers 不能小于 0")
    if args.chunk_size < 1:
        parser.error("--chunk-size 必须大于 0")
    if args.log_every < 1:
        parser.error("--log-every 必须大于 0")
    if args.plot_every < 0:
        parser.error("--plot-every 不能小于 0")
    if args.save_every < 0:
        parser.error("--save-every 不能小于 0")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours 必须大于 0")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm 必须大于 0")
    if args.episodes is not None and any(index < 0 for index in args.episodes):
        parser.error("episode index 不能为负数")
    return args


def resolve_device(requested: str) -> torch.device:
    """解析训练设备，并在显式请求不可用设备时尽早报错。"""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请改用 --device mps 或 --device cpu。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用；请改用 --device cuda 或 --device cpu。")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_delta_timestamps(indices: list[int] | None, fps: int) -> list[float]:
    """把离散帧偏移转换成 LeRobotDataset 使用的秒级偏移。"""
    if indices is None:
        return [0.0]
    return [index / fps for index in indices]


def split_policy_features(metadata: LeRobotDatasetMetadata):
    """从数据集元数据推导 ACT 的输入与输出特征。"""
    features = dataset_to_policy_features(metadata.features)
    output_features = {
        name: feature
        for name, feature in features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        name: feature
        for name, feature in features.items()
        if name not in output_features
    }
    if not output_features:
        raise ValueError("数据集中没有 ACT 可监督的 action 特征。")
    return input_features, output_features


def read_metrics(metrics_path: Path) -> list[dict[str, float]]:
    """读取训练指标，供恢复训练和绘图复用。"""
    if not metrics_path.is_file():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(metrics_file)
        ]


def prepare_metrics_file(metrics_path: Path, resume_step: int | None) -> None:
    """初始化 CSV；恢复时丢弃 checkpoint 之后可能残留的日志。"""
    rows = read_metrics(metrics_path) if resume_step is not None else []
    if resume_step is not None:
        rows = [row for row in rows if int(row["step"]) <= resume_step]
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(metrics_path: Path, row: dict[str, float]) -> None:
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writerow(row)


def plot_training_curves(metrics_path: Path, plot_path: Path) -> None:
    """从 CSV 生成适合教程直接引用的四联训练曲线。"""
    rows = read_metrics(metrics_path)
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(steps, [row["loss"] for row in rows], color="#1f77b4")
    axes[0, 0].set(title="ACT training loss", ylabel="Loss")

    axes[0, 1].plot(
        steps,
        [row["l1_loss"] for row in rows],
        label="L1 action loss",
        color="#2ca02c",
    )
    axes[0, 1].plot(
        steps,
        [row["kld_loss"] for row in rows],
        label="KL divergence",
        color="#ff7f0e",
    )
    axes[0, 1].set(title="Loss components", ylabel="Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [row["grad_norm"] for row in rows],
        color="#d62728",
    )
    axes[1, 0].set(title="Gradient norm before clipping", ylabel="L2 norm")

    axes[1, 1].plot(
        steps,
        [row["steps_per_s"] for row in rows],
        label="steps/s",
        color="#9467bd",
    )
    axes[1, 1].set(title="Training throughput", ylabel="Steps/s")
    memory_axis = axes[1, 1].twinx()
    memory_axis.plot(
        steps,
        [row["gpu_memory_gb"] for row in rows],
        label="GPU memory",
        color="#8c564b",
        alpha=0.75,
    )
    memory_axis.set_ylabel("Peak allocated GPU memory (GiB)")

    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)

    figure.suptitle("LeRobot ACT on ALOHA Transfer Cube", fontsize=14)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def save_policy_bundle(
    save_dir: Path,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


def save_resumable_checkpoint(
    checkpoint_dir: Path,
    step: int,
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    preprocessor,
    postprocessor,
) -> None:
    save_policy_bundle(checkpoint_dir, policy, preprocessor, postprocessor)
    save_training_state(checkpoint_dir, step, optimizer)
    update_last_checkpoint(checkpoint_dir)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp 当前只支持 CUDA；请去掉该参数或使用 --device cuda。")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    plot_path = output_dir / "training_curves.webp"
    summary_path = output_dir / "run_summary.json"
    resume_from = (
        args.resume_from.expanduser().resolve()
        if args.resume_from is not None
        else None
    )

    print(f"dataset: {args.dataset_id}@{args.revision}")
    print(f"device:  {device}")
    print(f"output:  {output_dir}")
    print(f"amp:     {args.amp}")

    metadata = LeRobotDatasetMetadata(
        args.dataset_id,
        revision=args.revision,
    )
    input_features, output_features = split_policy_features(metadata)

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        use_vae=not args.no_vae,
        kl_weight=args.kl_weight,
        push_to_hub=False,
    )
    if resume_from is None:
        policy = ACTPolicy(config).to(device)
    else:
        if not resume_from.is_dir():
            raise FileNotFoundError(f"checkpoint 不存在：{resume_from}")
        policy = ACTPolicy.from_pretrained(
            resume_from,
            config=config,
            local_files_only=True,
        ).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
    )

    delta_timestamps = {
        "action": make_delta_timestamps(
            config.action_delta_indices,
            metadata.fps,
        ),
        **{
            name: make_delta_timestamps(
                config.observation_delta_indices,
                metadata.fps,
            )
            for name in config.image_features
        },
    }
    dataset = LeRobotDataset(
        args.dataset_id,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
        revision=args.revision,
    )
    if len(dataset) < args.batch_size:
        raise ValueError(
            f"可用样本数 {len(dataset)} 小于 batch size {args.batch_size}；"
            "请减小 --batch-size 或加载更多 episode。",
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    step = 0
    if resume_from is not None:
        step, optimizer, _ = load_training_state(resume_from, optimizer, None)
        if step >= args.steps:
            raise ValueError(
                f"checkpoint 已在 step {step}，不小于目标 --steps {args.steps}。",
            )
        print(f"resume:  {resume_from} (step {step})")
    prepare_metrics_file(metrics_path, step if resume_from is not None else None)

    print(
        f"frames: {len(dataset)}, fps: {metadata.fps}, "
        f"chunk: {config.chunk_size}, vae: {config.use_vae}",
    )
    print(f"inputs:  {', '.join(input_features)}")
    print(f"outputs: {', '.join(output_features)}")

    policy.train()
    run_start = time.perf_counter()
    previous_log_time = run_start
    previous_log_step = step
    previous_rows = read_metrics(metrics_path)
    elapsed_offset = previous_rows[-1]["elapsed_s"] if previous_rows else 0.0
    metric_window: dict[str, list[float]] = {
        "loss": [],
        "l1_loss": [],
        "kld_loss": [],
        "grad_norm": [],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopped_by_time = False
    last_checkpoint_step: int | None = None

    while step < args.steps:
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch = preprocessor(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp,
            ):
                loss, loss_metrics = policy.forward(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()

            step += 1
            metric_window["loss"].append(loss.item())
            metric_window["grad_norm"].append(float(grad_norm))
            for name in ("l1_loss", "kld_loss"):
                value = loss_metrics.get(name, 0.0) if loss_metrics else 0.0
                metric_window[name].append(float(value))

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                now = time.perf_counter()
                log_steps = step - previous_log_step
                row = {
                    "step": float(step),
                    "elapsed_s": elapsed_offset + now - run_start,
                    "loss": mean(metric_window["loss"]),
                    "l1_loss": mean(metric_window["l1_loss"]),
                    "kld_loss": mean(metric_window["kld_loss"]),
                    "grad_norm": mean(metric_window["grad_norm"]),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "steps_per_s": log_steps / max(now - previous_log_time, 1e-9),
                    "gpu_memory_gb": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                append_metrics(metrics_path, row)
                print(
                    f"step={step:>6d} loss={row['loss']:.4f} "
                    f"l1={row['l1_loss']:.4f} kld={row['kld_loss']:.4f} "
                    f"grad_norm={row['grad_norm']:.4f} "
                    f"speed={row['steps_per_s']:.2f} step/s "
                    f"gpu={row['gpu_memory_gb']:.2f} GiB",
                    flush=True,
                )
                previous_log_time = now
                previous_log_step = step
                for values in metric_window.values():
                    values.clear()

            should_plot = (
                args.plot_every > 0
                and (step == 1 or step % args.plot_every == 0 or step == args.steps)
            )
            if should_plot:
                plot_training_curves(metrics_path, plot_path)

            should_checkpoint = (
                args.save_every > 0
                and (step % args.save_every == 0 or step == args.steps)
            )
            if should_checkpoint:
                width = max(6, len(str(args.steps)))
                checkpoint_dir = (
                    output_dir / "checkpoints" / f"step_{step:0{width}d}"
                )
                save_resumable_checkpoint(
                    checkpoint_dir,
                    step,
                    policy,
                    optimizer,
                    preprocessor,
                    postprocessor,
                )
                last_checkpoint_step = step
                print(f"checkpoint: {checkpoint_dir}", flush=True)

            if (
                args.max_hours is not None
                and (time.perf_counter() - run_start) >= args.max_hours * 3600
            ):
                stopped_by_time = True
                print(
                    f"已达到 {args.max_hours:.2f} 小时训练时限，正在安全保存。",
                    flush=True,
                )
                break

            if step >= args.steps:
                break

        if stopped_by_time:
            break

    if stopped_by_time and args.save_every > 0 and last_checkpoint_step != step:
        width = max(6, len(str(args.steps)))
        checkpoint_dir = output_dir / "checkpoints" / f"step_{step:0{width}d}"
        save_resumable_checkpoint(
            checkpoint_dir,
            step,
            policy,
            optimizer,
            preprocessor,
            postprocessor,
        )
        print(f"checkpoint: {checkpoint_dir}", flush=True)

    save_policy_bundle(output_dir, policy, preprocessor, postprocessor)
    save_training_state(output_dir, step, optimizer)
    plot_training_curves(metrics_path, plot_path)
    rows = read_metrics(metrics_path)
    summary = {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "episodes": args.episodes,
        "target_steps": args.steps,
        "completed_steps": step,
        "max_hours": args.max_hours,
        "stopped_by_time": stopped_by_time,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "use_vae": not args.no_vae,
        "amp": args.amp,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "wall_time_s": time.perf_counter() - run_start,
        "final_metrics": rows[-1] if rows else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint 已保存到：{output_dir}")
    print(f"训练指标：{metrics_path}")
    print(f"训练曲线：{plot_path}")
    return output_dir


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
