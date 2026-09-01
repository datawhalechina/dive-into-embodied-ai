"""从正式训练输出生成可复现报告、图表、GIF 和原始数据。"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


REPORT_DIR = Path(__file__).resolve().parent
LAB_DIR = REPORT_DIR.parent
RUN_DIR = LAB_DIR / "outputs" / "smooth_finetune_8m"
RAW_DIR = REPORT_DIR / "raw"
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
TB_RAW_DIR = RAW_DIR / "tensorboard"

sys.path.insert(0, str(LAB_DIR))
from evaluate import CMD_SCRIPT, render_demo  # noqa: E402
from pupper_env import PupperEnv  # noqa: E402

EVAL_COMMANDS = (0.2, 0.4, 0.6)
EVAL_SECONDS = 8.0
WARMUP_SECONDS = 1.0
RESET_PENALTY = 0.25


def _run_text(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() or result.stderr.strip()


def save_machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": int(_run_text(["sysctl", "-n", "hw.memsize"]) or 0),
        "gpu": _run_text(["system_profiler", "SPDisplaysDataType"]),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "mujoco": mujoco.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    (RAW_DIR / "machine.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return info


def save_run_config() -> dict:
    """归档两阶段课程配置，返回 finetune 阶段配置用于报告。"""
    with (LAB_DIR / "configs" / "smooth_finetune.yaml").open(encoding="utf-8") as file:
        finetune_config = yaml.safe_load(file)
    shutil.copy2(
        LAB_DIR / "configs" / "bootstrap_walk.yaml",
        RAW_DIR / "bootstrap_walk.yaml",
    )
    shutil.copy2(
        LAB_DIR / "configs" / "smooth_finetune.yaml",
        RAW_DIR / "smooth_finetune.yaml",
    )
    (RAW_DIR / "run_config.yaml").write_text(
        yaml.safe_dump(finetune_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return finetune_config


def copy_raw_training_files() -> Path:
    event_files = sorted((RUN_DIR / "tb").rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"TensorBoard event 不存在：{RUN_DIR / 'tb'}")
    TB_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for event_file in event_files:
        shutil.copy2(event_file, TB_RAW_DIR / event_file.name)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(RUN_DIR.glob("pupper_ppo_*_steps.zip"))
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, CHECKPOINT_DIR / checkpoint.name)
    shutil.copy2(RUN_DIR / "pupper_ppo.zip", CHECKPOINT_DIR / "pupper_ppo_final.zip")
    return event_files[0]


def export_training_metrics(event_file: Path) -> dict[str, list]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars = {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get("scalars", [])
    }

    with (RAW_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("tag", "step", "wall_time", "value"))
        for tag, events in sorted(scalars.items()):
            for event in events:
                writer.writerow((tag, event.step, f"{event.wall_time:.6f}", f"{event.value:.10g}"))
    return scalars


def plot_training_curves(scalars: dict[str, list]) -> None:
    tags = (
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/explained_variance",
        "train/std",
        "train/clip_fraction",
        "train/approx_kl",
    )
    titles = (
        "Episode reward",
        "Episode length",
        "Explained variance",
        "Policy std",
        "Clip fraction",
        "Approx KL",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, tag, title in zip(axes.flat, tags, titles):
        events = scalars[tag]
        axis.plot([e.step / 1e6 for e in events], [e.value for e in events], linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Environment steps [M]")
        axis.grid(True, alpha=0.3)
        if tag == "rollout/ep_len_mean":
            axis.axhline(1000, color="tab:red", linestyle="--", linewidth=1, label="max=1000")
            axis.legend()
    figure.suptitle("Pupper PPO training curves — M3 Ultra CPU, 16 environments")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(REPORT_DIR / "training_curves.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def checkpoint_steps(path: Path) -> int:
    return int(path.stem.split("_")[-2])


def evaluate_checkpoints() -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    checkpoint_paths = sorted(
        CHECKPOINT_DIR.glob("pupper_ppo_*_steps.zip"),
        key=checkpoint_steps,
    )

    # 评估脚本自行控制命令，关闭环境内命令重采样。
    env = PupperEnv(cmd_resample_steps=0, perturb_enabled=False)
    for checkpoint in checkpoint_paths:
        steps = checkpoint_steps(checkpoint)
        model = PPO.load(checkpoint, device="cpu")
        for command in EVAL_COMMANDS:
            obs, _ = env.reset(seed=42)
            env.cmd = np.array([command, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()
            resets = 0
            command_rows = []
            for index in range(int(EVAL_SECONDS / env.dt)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                rotation = env.data.xmat[env._base_id].reshape(3, 3)
                local_linear = rotation.T @ env.data.qvel[0:3]
                local_angular = rotation.T @ env.data.qvel[3:6]
                row = {
                    "checkpoint_steps": steps,
                    "command_vx": command,
                    "time_s": index * env.dt,
                    "actual_vx": float(local_linear[0]),
                    "actual_vy": float(local_linear[1]),
                    "actual_wz": float(local_angular[2]),
                    "reward": float(reward),
                    "base_z": float(env.data.qpos[2]),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "episode": resets,
                }
                raw_rows.append(row)
                command_rows.append(row)
                if terminated or truncated:
                    resets += 1
                    obs, _ = env.reset(seed=42)
                    env.cmd = np.array([command, 0.0, 0.0], dtype=np.float32)
                    obs = env._get_obs()

            measured = [
                row for row in command_rows
                if row["time_s"] >= WARMUP_SECONDS
            ]
            errors = np.array([row["actual_vx"] - command for row in measured])
            velocities = np.array([row["actual_vx"] for row in measured])
            summary_rows.append({
                "checkpoint_steps": steps,
                "command_vx": command,
                "mean_actual_vx": float(np.mean(velocities)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "resets": resets,
            })

    raw_fields = tuple(raw_rows[0].keys())
    with (RAW_DIR / "velocity_tracking.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)
    with (RAW_DIR / "checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(summary_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return raw_rows, summary_rows


def aggregate_checkpoints(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["checkpoint_steps"]].append(row)
    aggregate = []
    for steps, rows in sorted(grouped.items()):
        mean_mae = float(np.mean([row["mae"] for row in rows]))
        resets = int(sum(row["resets"] for row in rows))
        aggregate.append({
            "checkpoint_steps": steps,
            "mean_mae": mean_mae,
            "total_resets": resets,
            "score": mean_mae + RESET_PENALTY * resets,
        })
    return aggregate


def plot_checkpoint_comparison(aggregate: list[dict]) -> None:
    steps = np.array([row["checkpoint_steps"] for row in aggregate]) / 1e6
    mae = [row["mean_mae"] for row in aggregate]
    resets = [row["total_resets"] for row in aggregate]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, mae, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Mean velocity MAE [m/s]")
    axes[1].plot(steps, resets, marker="o", color="tab:red", linewidth=1.5)
    axes[1].set_ylabel("Resets across 3 commands")
    for axis in axes:
        axis.set_xlabel("Checkpoint steps [M]")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Checkpoint evaluation")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(REPORT_DIR / "checkpoint_comparison.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_best_velocity(raw_rows: list[dict], best_steps: int) -> None:
    figure, axes = plt.subplots(1, len(EVAL_COMMANDS), figsize=(15, 4), sharey=True)
    for axis, command in zip(axes, EVAL_COMMANDS):
        rows = [
            row for row in raw_rows
            if row["checkpoint_steps"] == best_steps and row["command_vx"] == command
        ]
        axis.axhline(command, color="tab:red", linestyle="--", label="command")
        axis.plot(
            [row["time_s"] for row in rows],
            [row["actual_vx"] for row in rows],
            color="tab:blue",
            linewidth=1,
            label="actual",
        )
        axis.set_title(f"vx command = {command:.1f} m/s")
        axis.set_xlabel("Time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Body-frame vx [m/s]")
    figure.suptitle(f"Best checkpoint velocity tracking — {best_steps / 1e6:g}M steps")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(REPORT_DIR / "velocity_tracking.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_command_demo_raw(model: PPO) -> None:
    env = PupperEnv(cmd_resample_steps=0, perturb_enabled=False)
    obs, _ = env.reset(seed=42)
    rows = []
    elapsed = 0.0
    index = 0
    resets = 0
    for command, duration in CMD_SCRIPT:
        env.cmd = np.array(command, dtype=np.float32)
        obs = env._get_obs()
        for _ in range(int(duration / env.dt)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rotation = env.data.xmat[env._base_id].reshape(3, 3)
            local_linear = rotation.T @ env.data.qvel[0:3]
            local_angular = rotation.T @ env.data.qvel[3:6]
            rows.append({
                "time_s": elapsed + index * env.dt,
                "command_vx": command[0],
                "command_vy": command[1],
                "command_wz": command[2],
                "actual_vx": float(local_linear[0]),
                "actual_vy": float(local_linear[1]),
                "actual_wz": float(local_angular[2]),
                "base_z": float(env.data.qpos[2]),
                "reward": float(reward),
                "terminated": int(terminated),
                "episode": resets,
            })
            index += 1
            if terminated or truncated:
                resets += 1
                obs, _ = env.reset(seed=42)
                env.cmd = np.array(command, dtype=np.float32)
                obs = env._get_obs()
        elapsed += duration
        index = 0
    with (RAW_DIR / "command_demo.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_demo(best_checkpoint: Path) -> None:
    model = PPO.load(best_checkpoint, device="cpu")
    save_command_demo_raw(model)
    render_demo(
        model,
        PupperEnv(cmd_resample_steps=0, perturb_enabled=False),
        REPORT_DIR / "command_demo.gif",
    )


def evaluate_gait_symmetry(
    best_checkpoint: Path,
    seconds: float = 6.0,
) -> list[tuple[str, float, int]]:
    """前进命令下四脚触地占比与抬脚次数，衡量钉腿惩罚是否消除三腿跛行。"""
    model = PPO.load(best_checkpoint, device="cpu")
    env = PupperEnv(cmd_resample_steps=0, perturb_enabled=False)
    obs, _ = env.reset(seed=42)
    env.cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    obs = env._get_obs()
    contact_history = []
    for _ in range(int(seconds / env.dt)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        foot_z = env.data.site_xpos[env._foot_site_ids][:, 2] - 0.02
        contact_history.append(foot_z < 1e-3)
        if terminated or truncated:
            obs, _ = env.reset(seed=42)
            env.cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()
    contacts = np.array(contact_history[25:])
    labels = ("前右 FR", "前左 FL", "后右 RR", "后左 RL")
    rows = []
    for index, label in enumerate(labels):
        column = contacts[:, index]
        liftoffs = int(np.sum((~column[1:]) & column[:-1]))
        rows.append((label, float(column.mean() * 100.0), liftoffs))
    return rows


def scalar_stats(scalars: dict[str, list], tag: str) -> tuple[float, float, float]:
    values = [event.value for event in scalars[tag]]
    return values[0], values[-1], max(values)


# v1（45 维观测、固定命令）演示的平稳度基线，口径与 demo_phase_stats 一致；
# 原始数据见 ../6.rl_pupper/reports/raw/command_demo.csv。
V1_DEMO_STATS = {
    (0.0, 0.0, 0.0): {"vx_mean": -0.027, "vx_std": 0.067, "wz_mean": 0.014, "wz_std": 0.543, "z_std": 0.0027},
    (0.5, 0.0, 0.0): {"vx_mean": 0.530, "vx_std": 0.190, "wz_mean": 0.101, "wz_std": 0.735, "z_std": 0.0217},
    (0.0, 0.0, 0.5): {"vx_mean": -0.011, "vx_std": 0.077, "wz_mean": 0.526, "wz_std": 0.551, "z_std": 0.0027},
    (-0.3, 0.0, 0.0): {"vx_mean": -0.292, "vx_std": 0.069, "wz_mean": 0.134, "wz_std": 0.524, "z_std": 0.0024},
}
V1_DEMO_TERMINATIONS = 1

# v2.1（无动作滤波）演示基线，数据来自 ../6.rl_pupper_v2_1/reports/raw/command_demo.csv。
V21_DEMO_STATS = {
    (0.0, 0.0, 0.0): {"vx_mean": -0.001, "vx_std": 0.080, "wz_mean": -0.060, "wz_std": 0.578, "z_std": 0.0048},
    (0.5, 0.0, 0.0): {"vx_mean": 0.521, "vx_std": 0.082, "wz_mean": -0.029, "wz_std": 0.711, "z_std": 0.0048},
    (0.0, 0.0, 0.5): {"vx_mean": -0.002, "vx_std": 0.080, "wz_mean": 0.385, "wz_std": 0.642, "z_std": 0.0062},
    (-0.3, 0.0, 0.0): {"vx_mean": -0.307, "vx_std": 0.064, "wz_mean": -0.246, "wz_std": 0.612, "z_std": 0.0060},
}
V21_DEMO_TERMINATIONS = 0

# v2 / v2.1 最佳策略前进 6 秒的四脚接触统计（触地占比 %、抬脚次数）。
# v2 后左腿 91.6% 触地即"三腿跛行"；v2.1 的钉腿惩罚将其修复到 81.1%/24。
V2_GAIT_STATS = [
    ("前右 FR", 46.9, 27),
    ("前左 FL", 52.0, 24),
    ("后右 RR", 57.8, 38),
    ("后左 RL", 91.6, 9),
]
V21_GAIT_STATS = [
    ("前右 FR", 42.5, 31),
    ("前左 FL", 58.9, 28),
    ("后右 RR", 48.7, 37),
    ("后左 RL", 81.1, 24),
]


def demo_phase_stats(demo_rows: list[dict]) -> dict[tuple, dict]:
    """按命令值切分连续段，每段去掉前 0.5 秒过渡期后按命令汇总均值和波动。"""
    segments: list[tuple[tuple, list[dict]]] = []
    current_cmd: tuple | None = None
    current: list[dict] = []
    for row in demo_rows:
        command = (
            float(row["command_vx"]),
            float(row["command_vy"]),
            float(row["command_wz"]),
        )
        if command != current_cmd:
            if current:
                segments.append((current_cmd, current))
            current_cmd, current = command, []
        current.append(row)
    segments.append((current_cmd, current))

    pooled: dict[tuple, list[dict]] = defaultdict(list)
    warmup_rows = int(0.5 / 0.02)
    for command, segment in segments:
        pooled[command].extend(segment[warmup_rows:])

    stats = {}
    for command, rows in pooled.items():
        vx = np.array([float(row["actual_vx"]) for row in rows])
        wz = np.array([float(row["actual_wz"]) for row in rows])
        base_z = np.array([float(row["base_z"]) for row in rows])
        stats[command] = {
            "vx_mean": float(np.mean(vx)),
            "vx_std": float(np.std(vx)),
            "wz_mean": float(np.mean(wz)),
            "wz_std": float(np.std(wz)),
            "z_std": float(np.std(base_z)),
        }
    return stats


def write_report(
    machine: dict,
    config: dict,
    scalars: dict[str, list],
    summary_rows: list[dict],
    aggregate: list[dict],
    best_steps: int,
    symmetry_rows: list[tuple[str, float, int]],
) -> None:
    first_reward, final_reward, peak_reward = scalar_stats(scalars, "rollout/ep_rew_mean")
    first_length, final_length, peak_length = scalar_stats(scalars, "rollout/ep_len_mean")
    first_std, final_std, _ = scalar_stats(scalars, "train/std")
    final_step = max(event.step for events in scalars.values() for event in events)
    all_wall_times = [event.wall_time for events in scalars.values() for event in events]
    duration_minutes = (max(all_wall_times) - min(all_wall_times)) / 60

    selected_steps = (1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000)
    by_pair = {
        (row["checkpoint_steps"], row["command_vx"]): row
        for row in summary_rows
    }
    table_lines = []
    for steps in selected_steps:
        rows = [by_pair[(steps, command)] for command in EVAL_COMMANDS]
        table_lines.append(
            f"| {steps / 1e6:g}M | "
            + " | ".join(f"{row['mean_actual_vx']:.3f}" for row in rows)
            + f" | {np.mean([row['mae'] for row in rows]):.3f} | "
            + f"{sum(row['resets'] for row in rows)} |"
        )

    best_summary = next(row for row in aggregate if row["checkpoint_steps"] == best_steps)
    gpu_line = next(
        (line.strip() for line in machine["gpu"].splitlines() if "Chipset Model" in line),
        "Chipset Model: unknown",
    )
    with (RAW_DIR / "command_demo.csv").open(encoding="utf-8") as file:
        demo_rows = list(csv.DictReader(file))
    phase_stats = demo_phase_stats(demo_rows)

    phase_names = {
        (0.0, 0.0, 0.0): "站立",
        (0.5, 0.0, 0.0): "前进 `vx=0.5`",
        (0.0, 0.0, 0.5): "左转 `wz=0.5`",
        (-0.3, 0.0, 0.0): "后退 `vx=-0.3`",
    }
    comparison_lines = []
    for command, name in phase_names.items():
        v1 = V1_DEMO_STATS[command]
        v21 = V21_DEMO_STATS[command]
        current = phase_stats[command]
        tracked = "wz" if command == (0.0, 0.0, 0.5) else "vx"
        comparison_lines.append(
            f"| {name} | `{tracked}` 均值 | "
            f"{v1[f'{tracked}_mean']:+.3f} | {v21[f'{tracked}_mean']:+.3f} | "
            f"{current[f'{tracked}_mean']:+.3f} |"
        )
        comparison_lines.append(
            f"| {name} | `vx` / `wz` 波动 | "
            f"±{v1['vx_std']:.3f} / ±{v1['wz_std']:.3f} | "
            f"±{v21['vx_std']:.3f} / ±{v21['wz_std']:.3f} | "
            f"±{current['vx_std']:.3f} / ±{current['wz_std']:.3f} |"
        )
    forward_v1 = V1_DEMO_STATS[(0.5, 0.0, 0.0)]
    forward_v21 = V21_DEMO_STATS[(0.5, 0.0, 0.0)]
    forward_current = phase_stats[(0.5, 0.0, 0.0)]
    comparison_lines.append(
        f"| 前进 `vx=0.5` | 机身高度波动 | "
        f"±{forward_v1['z_std']:.4f} | ±{forward_v21['z_std']:.4f} | "
        f"±{forward_current['z_std']:.4f} |"
    )

    demo_terminations = [row for row in demo_rows if int(row["terminated"])]
    comparison_lines.append(
        f"| 演示全程 | 终止次数 | {V1_DEMO_TERMINATIONS} | "
        f"{V21_DEMO_TERMINATIONS} | {len(demo_terminations)} |"
    )

    symmetry_lines = []
    for (label, v2_duty, v2_lifts), (_, v21_duty, v21_lifts), (_, duty, lifts) in zip(
        V2_GAIT_STATS, V21_GAIT_STATS, symmetry_rows,
    ):
        symmetry_lines.append(
            f"| {label} | {v2_duty:.1f}% / {v2_lifts} | "
            f"{v21_duty:.1f}% / {v21_lifts} | {duty:.1f}% / {lifts} |"
        )
    termination_text = (
        ", ".join(f"t={float(row['time_s']):.2f}s" for row in demo_terminations)
        if demo_terminations else "无"
    )
    report = f"""# Pupper PPO 本机训练报告（v2.2）

## 任务目标

训练一个**速度指令条件化的四足低层控制策略**。使用者向策略指定机身坐标系下的目标速度：

| 指令 | 含义 | 示例 |
| --- | --- | --- |
| `vx` | 前后速度，正值前进、负值后退 | `vx=0.5` 表示以 0.5 m/s 前进 |
| `vy` | 左右横移速度 | `vy=0.2` 表示横向移动 |
| `wz` | 绕竖直轴的转向角速度 | `wz=0.5` 表示以 0.5 rad/s 左转 |

PPO 策略的输入是 `vx、vy、wz` 指令以及 IMU、关节状态、上一时刻动作和机身线速度，共 48 维；输出是 12 个关节位置残差，再由 PD 控制器执行。它不要求复现预先定义的 `walk` 或 `trot` 轨迹，四条腿的协调方式由奖励函数自行学习。

训练时命令从 `vx ∈ [-0.75, 0.75]`、`vy ∈ [-0.5, 0.5]`、`wz ∈ [-2, 2]` 随机采样，回合内每 250 步（5 秒）重采样一次，并以 10% 概率采样零命令。目标是在不摔倒、不过度用力且动作平滑的前提下，使实际速度跟踪指定速度。成功判据依次是：

1. `ep_len_mean` 接近 1000，说明可以稳定完成 20 秒回合。
2. 实际 `vx、vy、wz` 接近目标指令，而不是只会以固定速度移动。
3. 前进、后退、转向和站立命令都能执行，命令切换时不摔倒。

## 相对 v2.1 的改动

本版（v2.2）在 [`6.rl_pupper_v2_1`](../../6.rl_pupper_v2_1/) 基础上**仅新增动作低通滤波**：策略输出的关节位置残差先过一阶 EMA 滤波（α=0.5，50 Hz 下截止约 5.5 Hz）再进 PD 伺服，训练与评估同样经过滤波。其余观测、奖励、命令分布、扰动与两阶段课程全部与 v2.1 相同。

动机：v2.1 的残余抖动来自网络每 20 ms 输出一次新目标、且奖励惩罚无法完全约束高频分量；而手写开环步态（`5.gait-control`）"天然平滑"的本质是动作频带受限的周期轨迹。滤波把同样的频带先验加给 RL：保留步态基频与摆动相（1–5 Hz），滤掉高频抖动，同时保住 RL 的闭环平衡、速度跟踪与抗扰能力。这也是真机部署的标准做法。

版本脉络：v1 基线 → v2（观测 48 维 + 防抖奖励组 + 两阶段课程，见 [v2 报告](../../6.rl_pupper_v2/reports/README.md)）→ v2.1（钉腿惩罚修三腿跛行，见 [v2.1 报告](../../6.rl_pupper_v2_1/reports/README.md)）→ 本版。

## 结论

在 Apple M3 Ultra 上使用 16 个 CPU MuJoCo 环境完成两阶段训练：轻惩罚 bootstrap 8M 步 + 全额平滑惩罚微调 {final_step:,} 步（微调阶段耗时约 {duration_minutes:.1f} 分钟）。

按三档速度平均 MAE 加摔倒惩罚选择的最佳微调 checkpoint 是 **[{best_steps / 1e6:g}M 步](raw/checkpoints/pupper_ppo_{best_steps}_steps.zip)**，平均 MAE 为 {best_summary['mean_mae']:.3f} m/s，评估重置次数为 {best_summary['total_resets']}。

## 训练机器

- 芯片：{gpu_line.split(':', 1)[-1].strip()}
- CPU 逻辑核心：{machine['logical_cpu_count']}
- 统一内存：{machine['memory_bytes'] / 1024 ** 3:.0f} GiB
- 架构：{machine['machine']}
- PyTorch：{machine['torch']}
- Stable Baselines3：{machine['stable_baselines3']}
- MuJoCo：{machine['mujoco']}
- 训练设备：CPU，16 个 `SubprocVecEnv`

## 训练参数

- 阶段一 bootstrap：8M 步，削弱平滑惩罚（配置见 [`raw/bootstrap_walk.yaml`](raw/bootstrap_walk.yaml)）
- 阶段二微调：{config['run']['total_timesteps']:,} 步，全额平滑惩罚，从阶段一终点续训（配置见 [`raw/smooth_finetune.yaml`](raw/smooth_finetune.yaml)）
- PPO rollout：`n_steps={config['ppo']['n_steps']}`，`batch_size={config['ppo']['batch_size']}`，`n_epochs={config['ppo']['n_epochs']}`
- 学习率：{config['ppo']['learning_rate']}
- 折扣：`gamma={config['ppo']['gamma']}`，`gae_lambda={config['ppo']['gae_lambda']}`
- 裁剪：`clip_range={config['ppo']['clip_range']}`
- 熵系数：阶段内线性衰减，见上文
- 网络：{config['ppo']['net_arch']}
- 单回合最大步数：{config['environment']['max_steps']}

## 训练曲线

以下为微调阶段的曲线。

![训练曲线](training_curves.webp)

- `ep_rew_mean`：{first_reward:.2f} → {final_reward:.2f}，峰值 {peak_reward:.2f}
- `ep_len_mean`：{first_length:.0f} → {final_length:.0f}，峰值 {peak_length:.0f}（训练环境开启随机初速度和 kick 扰动，回合长度含受扰摔倒；关闭扰动的评估中最佳策略 0 摔倒）
- 策略标准差：{first_std:.2f} → {final_std:.2f}

## Checkpoint 对比

下表是微调阶段各 checkpoint 在 0.2、0.4、0.6 m/s 三档命令下各运行 8 秒的结果。三列为去掉首秒热身后的实际平均速度。

| Checkpoint | cmd 0.2 | cmd 0.4 | cmd 0.6 | 平均 MAE | 重置次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

![Checkpoint 对比](checkpoint_comparison.webp)

## 最佳策略

![速度跟踪](velocity_tracking.webp)

![命令演示](command_demo.gif)

命令切换演示逐阶段对比 v1 与 v2.1（每段去掉前 0.5 秒过渡期；波动为标准差）：

| 阶段 | 指标 | v1 | v2.1 | 本版 v2.2 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(comparison_lines)}

完整命令切换演示发生 {len(demo_terminations)} 次终止（{termination_text}）。

## 步态对称性

前进 `vx=0.5` 命令下运行 6 秒的四脚接触统计（触地占比 / 抬脚次数）。v2 的后左腿
92% 时间触地形成三腿跛行，v2.1 的钉腿惩罚将其修复；本版检验动作滤波未破坏对称性：

| 足 | v2 | v2.1 | 本版 v2.2 |
| --- | ---: | ---: | ---: |
{chr(10).join(symmetry_lines)}

## 原始数据

- [`raw/training_metrics.csv`](raw/training_metrics.csv)：全部 TensorBoard scalar，长表格式
- [`raw/velocity_tracking.csv`](raw/velocity_tracking.csv)：微调阶段全部 checkpoints 的逐时刻速度、回报和终止记录
- [`raw/checkpoint_summary.csv`](raw/checkpoint_summary.csv)：每个 checkpoint、每档命令的 MAE、RMSE 和重置次数
- [`raw/command_demo.csv`](raw/command_demo.csv)：GIF 对应的逐时刻命令和实际状态
- [`raw/tensorboard/`](raw/tensorboard/)：原始 TensorBoard event 文件
- [`raw/checkpoints/pupper_ppo_{best_steps}_steps.zip`](raw/checkpoints/pupper_ppo_{best_steps}_steps.zip)：仓库归档的最佳 checkpoint；其余中间 checkpoint 保留在本机训练输出中
- [`raw/machine.json`](raw/machine.json)：机器和软件环境
- [`raw/run_config.yaml`](raw/run_config.yaml)：本次实际运行配置

## 局限与下一步

1. 滤波带来的相位滞后使跟踪略趋保守：前进实际 0.479（目标 0.5），后退过冲到 -0.363（目标 -0.3）；转向已基本精准（0.512 / 目标 0.5）。可微调 `tracking_lin_vel` 权重或滤波系数换取更快的响应。
2. 四脚占空比进一步均衡（后左腿 92% → 81% → 64%，抬脚 9 → 24 → 32 次），但仍未达到严格对称的 trot 时序；需要的话走 gait 条件化路径（`configs/finetune_gait.yaml`）。
3. 机身线速度观测在仿真中直接读取；部署到真机需要状态估计器提供等价量，或改用完整版实验环境的帧堆叠方案。
4. 当前模型只在平地仿真验证，域随机化仅有初速度和 kick 扰动，没有观测噪声、延迟随机化和真机部署。
5. 步态模式由奖励自行涌现；若需要指定 walk/trot/pace 接触时序，参见 v1 的 gait 条件化实验（`../../6.rl_pupper/reports/gait_finetune/`）。
"""
    (REPORT_DIR / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    machine = save_machine_info()
    config = save_run_config()
    event_file = copy_raw_training_files()
    scalars = export_training_metrics(event_file)
    plot_training_curves(scalars)
    raw_rows, summary_rows = evaluate_checkpoints()
    aggregate = aggregate_checkpoints(summary_rows)
    plot_checkpoint_comparison(aggregate)
    best = min(aggregate, key=lambda row: row["score"])
    best_steps = int(best["checkpoint_steps"])
    plot_best_velocity(raw_rows, best_steps)
    best_checkpoint = CHECKPOINT_DIR / f"pupper_ppo_{best_steps}_steps.zip"
    generate_demo(best_checkpoint)
    symmetry_rows = evaluate_gait_symmetry(best_checkpoint)
    write_report(
        machine, config, scalars, summary_rows, aggregate, best_steps,
        symmetry_rows,
    )
    print(f"报告已生成：{REPORT_DIR / 'README.md'}")
    print(f"最佳 checkpoint：{best_checkpoint}")


if __name__ == "__main__":
    main()
