"""在 AMD ROCm GPU 上使用 LeRobotDataset 训练 ACT。

这是 ``codes/practices/vla/act/train_act.py`` 的 AMD Linux 版本。PyTorch
的 ROCm 后端沿用 ``torch.cuda`` API；本脚本会验证当前 wheel 确实包含
HIP，并在开始下载数据前执行 GPU 张量自检，绝不静默回退到 CPU。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_OUTPUT_DIR = LAB_DIR / "outputs" / "act_aloha_transfer_rocm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 AMD ROCm GPU 上用 ALOHA 仿真示教训练 LeRobot ACT。",
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
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="ResNet backbone 学习率；默认与 --learning-rate 相同。",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="关闭 ACT 的 CVAE，便于做消融实验。",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="ROCm GPU 的可见设备序号；默认使用 cuda:0。",
    )
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav"),
        default="pyav",
        help="视频解码后端；两者都在 CPU 上解码，再由预处理器搬到 GPU。",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="训练精度；支持 BF16 的 AMD GPU 可用 bf16 提速。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=10,
        help="每隔多少 step 向 training_metrics.csv 写一行。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10_000,
        help="滚动保存 training_state.pt 的间隔；设为 0 可关闭。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output-dir/training_state.pt 恢复模型、优化器和 step。",
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
    if args.learning_rate <= 0:
        parser.error("--learning-rate 必须大于 0")
    if args.backbone_learning_rate is not None and args.backbone_learning_rate <= 0:
        parser.error("--backbone-learning-rate 必须大于 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay 不能小于 0")
    if args.device_index < 0:
        parser.error("--device-index 不能小于 0")
    if args.log_every < 1:
        parser.error("--log-every 必须大于 0")
    if args.metrics_every < 1:
        parser.error("--metrics-every 必须大于 0")
    if args.save_every < 0:
        parser.error("--save-every 不能小于 0")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm 必须大于 0")
    if args.episodes is not None and any(index < 0 for index in args.episodes):
        parser.error("episode index 不能为负数")
    return args


def require_rocm_device(device_index: int = 0) -> torch.device:
    """返回可用的 ROCm 设备；缺少 HIP/AMD GPU 时立即失败。"""
    if torch.version.hip is None:
        raise RuntimeError(
            "当前 PyTorch 不含 ROCm/HIP；请在本目录执行 uv sync --frozen。",
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch ROCm 已安装，但未发现可用 AMD GPU；训练不会回退到 CPU。",
        )

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            f"请求的 ROCm 设备 cuda:{device_index} 不存在；"
            f"当前只有 {device_count} 个可见设备。",
        )

    # ROCm 在 PyTorch 中复用 cuda 设备名，"rocm" 和 "hip" 不是合法设备名。
    device = torch.device("cuda", device_index)
    try:
        probe = torch.arange(16, dtype=torch.float32, device=device)
        result = ((probe + 1) * 2).sum().item()
        if result != 272.0:
            raise RuntimeError(f"ROCm 张量计算结果异常：{result}")
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"ROCm GPU cuda:{device_index} 自检失败；训练不会回退到 CPU：{exc}",
        ) from exc

    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def json_serializable_args(args: argparse.Namespace) -> dict:
    """把 argparse 参数转换成适合保存到 JSON 的值。"""
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def save_training_state(
    path: Path,
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
    elapsed_seconds: float,
) -> None:
    """原子更新滚动训练状态，供长时间 ROCm 训练断点续跑。"""
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "rocm_random_state": torch.cuda.get_rng_state_all(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def restore_training_state(
    path: Path,
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """恢复模型、优化器和随机数状态，返回已完成 step 与累计用时。"""
    state = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    torch.cuda.set_rng_state_all(
        [random_state.cpu() for random_state in state["rocm_random_state"]],
    )
    return int(state["step"]), float(state.get("elapsed_seconds", 0.0))


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = require_rocm_device(args.device_index)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.csv"
    state_path = output_dir / "training_state.pt"
    run_config_path = output_dir / "training_run.json"
    summary_path = output_dir / "training_summary.json"

    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"找不到可恢复的训练状态：{state_path}")
    if not args.resume and metrics_path.exists():
        raise FileExistsError(
            f"输出目录已有训练指标：{metrics_path}；"
            "请更换 --output-dir，或使用 --resume 继续。",
        )

    print(f"dataset: {args.dataset_id}@{args.revision}")
    print(
        f"device:  {device} / {torch.cuda.get_device_name(device)} "
        f"(ROCm {torch.version.hip}, PyTorch {torch.__version__})",
    )
    print(f"output:  {output_dir}")

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
        optimizer_lr=args.learning_rate,
        optimizer_lr_backbone=(
            args.backbone_learning_rate
            if args.backbone_learning_rate is not None
            else args.learning_rate
        ),
        optimizer_weight_decay=args.weight_decay,
        push_to_hub=False,
    )
    policy = ACTPolicy(config).to(device)
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
        video_backend=args.video_backend,
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
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    optimizer = config.get_optimizer_preset().build(policy.parameters())

    start_step = 0
    elapsed_offset = 0.0
    if args.resume:
        start_step, elapsed_offset = restore_training_state(
            state_path,
            policy,
            optimizer,
            device,
        )
        if start_step >= args.steps:
            raise ValueError(
                f"checkpoint 已完成 {start_step} step，不小于目标 {args.steps}。",
            )
        print(f"从 step {start_step} 恢复训练：{state_path}")
    else:
        run_config_path.write_text(
            json.dumps(
                {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "arguments": json_serializable_args(args),
                    "dataset_frames": len(dataset),
                    "dataset_fps": metadata.fps,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "gpu": torch.cuda.get_device_name(device),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"frames: {len(dataset)}, fps: {metadata.fps}, "
        f"chunk: {config.chunk_size}, vae: {config.use_vae}, "
        f"video: {args.video_backend}, precision: {args.precision}",
    )
    print(f"inputs:  {', '.join(input_features)}")
    print(f"outputs: {', '.join(output_features)}")

    policy.train()
    torch.cuda.reset_peak_memory_stats(device)
    historical_peak_memory_gb = 0.0
    if args.resume and metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as existing_metrics:
            historical_peak_memory_gb = max(
                (
                    float(row["max_gpu_memory_gb"])
                    for row in csv.DictReader(existing_metrics)
                ),
                default=0.0,
            )
    step = start_step
    last_loss = float("nan")
    training_started = time.perf_counter()
    metrics_fields = (
        "step",
        "loss",
        "l1_loss",
        "kld_loss",
        "grad_norm",
        "learning_rate",
        "elapsed_seconds",
        "steps_per_second",
        "max_gpu_memory_gb",
    )
    metrics_mode = "a" if args.resume else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8", newline="", buffering=1) as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        if not args.resume:
            metrics_writer.writeheader()

        while step < args.steps:
            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                batch = preprocessor(batch)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else nullcontext()
                )
                with autocast_context:
                    loss, loss_metrics = policy.forward(batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()

                step += 1
                last_loss = loss.item()
                now = time.perf_counter()
                session_elapsed = now - training_started
                elapsed_seconds = elapsed_offset + session_elapsed
                # 使用累计吞吐，断点续训后的首个采样点不会跌成“1 / 启动耗时”。
                steps_per_second = step / max(elapsed_seconds, 1e-9)

                should_log = step == start_step + 1 or step % args.log_every == 0 or step == args.steps
                should_record = (
                    step == start_step + 1
                    or step % args.metrics_every == 0
                    or step == args.steps
                )
                if should_record:
                    metrics_writer.writerow(
                        {
                            "step": step,
                            "loss": f"{last_loss:.8f}",
                            "l1_loss": f"{loss_metrics.get('l1_loss', float('nan')):.8f}",
                            "kld_loss": f"{loss_metrics.get('kld_loss', float('nan')):.8f}",
                            "grad_norm": f"{float(grad_norm):.8f}",
                            "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
                            "elapsed_seconds": f"{elapsed_seconds:.3f}",
                            "steps_per_second": f"{steps_per_second:.5f}",
                            "max_gpu_memory_gb": f"{max(historical_peak_memory_gb, torch.cuda.max_memory_allocated(device) / 2**30):.4f}",
                        },
                    )

                if should_log:
                    metric_text = ""
                    if loss_metrics:
                        metric_text = " " + " ".join(
                            f"{name}={value:.4f}"
                            for name, value in loss_metrics.items()
                            if isinstance(value, (int, float))
                        )
                    print(
                        f"step={step:>6d} loss={last_loss:.4f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"speed={steps_per_second:.2f}step/s{metric_text}",
                    )

                if args.save_every and step % args.save_every == 0:
                    print(f"保存滚动训练状态：step {step}")
                    save_training_state(
                        state_path,
                        policy,
                        optimizer,
                        step,
                        elapsed_seconds,
                    )

                if step >= args.steps:
                    break

    total_elapsed = elapsed_offset + (time.perf_counter() - training_started)
    save_training_state(
        state_path,
        policy,
        optimizer,
        step,
        total_elapsed,
    )

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    summary_path.write_text(
        json.dumps(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "steps": step,
                "final_loss": last_loss,
                "elapsed_seconds": total_elapsed,
                "average_steps_per_second": step / max(total_elapsed, 1e-9),
                "max_gpu_memory_gb": max(
                    historical_peak_memory_gb,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint 已保存到：{output_dir}")
    return output_dir


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
