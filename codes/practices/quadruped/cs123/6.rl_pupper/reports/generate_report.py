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
RUN_DIR = LAB_DIR / "outputs" / "m3_ultra_cpu_20m"
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
    with (LAB_DIR / "configs" / "train.yaml").open(encoding="utf-8") as file:
        base_config = yaml.safe_load(file)
    shutil.copy2(LAB_DIR / "configs" / "train.yaml", RAW_DIR / "base_train.yaml")

    actual = base_config.copy()
    actual["run"] = dict(base_config["run"])
    actual["run"].update({
        "total_timesteps": 20_000_000,
        "n_envs": 16,
        "device": "cpu",
        "tensorboard": True,
        "output_dir": "outputs/m3_ultra_cpu_20m",
        "resume_from": None,
    })
    (RAW_DIR / "run_config.yaml").write_text(
        yaml.safe_dump(actual, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return actual


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

    env = PupperEnv()
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
    env = PupperEnv()
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
    render_demo(model, PupperEnv(), REPORT_DIR / "command_demo.gif")


def scalar_stats(scalars: dict[str, list], tag: str) -> tuple[float, float, float]:
    values = [event.value for event in scalars[tag]]
    return values[0], values[-1], max(values)


def write_report(
    machine: dict,
    config: dict,
    scalars: dict[str, list],
    summary_rows: list[dict],
    aggregate: list[dict],
    best_steps: int,
) -> None:
    first_reward, final_reward, peak_reward = scalar_stats(scalars, "rollout/ep_rew_mean")
    first_length, final_length, peak_length = scalar_stats(scalars, "rollout/ep_len_mean")
    final_step = max(event.step for events in scalars.values() for event in events)
    all_wall_times = [event.wall_time for events in scalars.values() for event in events]
    duration_minutes = (max(all_wall_times) - min(all_wall_times)) / 60

    selected_steps = (1_000_000, 2_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000)
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
    demo_groups: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for row in demo_rows:
        key = (
            float(row["command_vx"]),
            float(row["command_vy"]),
            float(row["command_wz"]),
        )
        demo_groups[key].append(row)

    def demo_mean(command: tuple[float, float, float], field: str) -> float:
        rows = demo_groups[command][25:]
        return float(np.mean([float(row[field]) for row in rows]))

    demo_terminations = [row for row in demo_rows if int(row["terminated"])]
    termination_text = (
        ", ".join(f"t={float(row['time_s']):.2f}s" for row in demo_terminations)
        if demo_terminations else "无"
    )
    report = f"""# Pupper PPO 本机训练报告

## 任务目标

训练一个**速度指令条件化的四足低层控制策略**。使用者向策略指定机身坐标系下的目标速度：

| 指令 | 含义 | 示例 |
| --- | --- | --- |
| `vx` | 前后速度，正值前进、负值后退 | `vx=0.5` 表示以 0.5 m/s 前进 |
| `vy` | 左右横移速度 | `vy=0.2` 表示横向移动 |
| `wz` | 绕竖直轴的转向角速度 | `wz=0.5` 表示以 0.5 rad/s 左转 |

PPO 策略的输入是 `vx、vy、wz` 指令以及 IMU、关节状态和上一时刻动作，共 45 维；输出是 12 个关节位置残差，再由 PD 控制器执行。它不要求复现预先定义的 `walk` 或 `trot` 轨迹，四条腿的协调方式由奖励函数自行学习。

训练时每个回合随机采样一组 `vx ∈ [-0.75, 0.75]`、`vy ∈ [-0.5, 0.5]`、`wz ∈ [-2, 2]`，目标是在不摔倒、不过度用力且动作平滑的前提下，使实际速度跟踪指定速度。成功判据依次是：

1. `ep_len_mean` 接近 1000，说明可以稳定完成 20 秒回合。
2. 实际 `vx、vy、wz` 接近目标指令，而不是只会以固定速度移动。
3. 前进、后退、转向和站立命令都能执行，命令切换时不摔倒。

当前精简环境在单个训练回合内不会切换命令，因此第 3 项中的动态切换属于额外泛化测试。

## 结论

在 Apple M3 Ultra 上使用 16 个 CPU MuJoCo 环境完成 {final_step:,} 步 PPO 训练，耗时约 {duration_minutes:.1f} 分钟。策略从频繁摔倒发展为可稳定前进，并开始区分速度命令；但低速命令仍明显超调，尚未达到准确速度跟踪。

按三档速度平均 MAE 加摔倒惩罚选择的最佳 checkpoint 是 **[{best_steps / 1e6:g}M 步](raw/checkpoints/pupper_ppo_{best_steps}_steps.zip)**，平均 MAE 为 {best_summary['mean_mae']:.3f} m/s，评估重置次数为 {best_summary['total_resets']}。

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

- 总步数：{config['run']['total_timesteps']:,}
- PPO rollout：`n_steps={config['ppo']['n_steps']}`，`batch_size={config['ppo']['batch_size']}`，`n_epochs={config['ppo']['n_epochs']}`
- 学习率：{config['ppo']['learning_rate']}
- 折扣：`gamma={config['ppo']['gamma']}`，`gae_lambda={config['ppo']['gae_lambda']}`
- 裁剪：`clip_range={config['ppo']['clip_range']}`
- 熵系数：`ent_coef={config['ppo']['ent_coef']}`
- 网络：{config['ppo']['net_arch']}
- 单回合最大步数：{config['environment']['max_steps']}

完整配置见 [`raw/run_config.yaml`](raw/run_config.yaml)。

## 训练曲线

![训练曲线](training_curves.webp)

- `ep_rew_mean`：{first_reward:.2f} → {final_reward:.2f}，峰值 {peak_reward:.2f}
- `ep_len_mean`：{first_length:.0f} → {final_length:.0f}，峰值 {peak_length:.0f}，未稳定饱和到 1000
- 探索标准差持续上升，表明 `ent_coef=0.01` 对该精简环境可能偏强

## Checkpoint 对比

下表是在 0.2、0.4、0.6 m/s 三档命令下各运行 8 秒的结果。三列为去掉首秒热身后的实际平均速度。

| Checkpoint | cmd 0.2 | cmd 0.4 | cmd 0.6 | 平均 MAE | 重置次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

![Checkpoint 对比](checkpoint_comparison.webp)

## 最佳策略

![速度跟踪](velocity_tracking.webp)

![命令演示](command_demo.gif)

完整命令切换序列的均值：

| 命令 | 目标 | 实际均值 |
| --- | ---: | ---: |
| 站立 | `vx=0, wz=0` | `vx={demo_mean((0.0, 0.0, 0.0), 'actual_vx'):.3f}, wz={demo_mean((0.0, 0.0, 0.0), 'actual_wz'):.3f}` |
| 前进 | `vx=0.5` | `vx={demo_mean((0.5, 0.0, 0.0), 'actual_vx'):.3f}` |
| 左转 | `wz=0.5` | `wz={demo_mean((0.0, 0.0, 0.5), 'actual_wz'):.3f}` |
| 后退 | `vx=-0.3` | `vx={demo_mean((-0.3, 0.0, 0.0), 'actual_vx'):.3f}` |

固定速度评估中最佳策略没有摔倒，但完整命令切换演示发生 {len(demo_terminations)} 次终止（{termination_text}）。它已经学会“稳定前进、转向、后退和粗粒度变速”，尚未学会高精度低速控制和完全稳定的命令切换。

## 原始数据

- [`raw/training_metrics.csv`](raw/training_metrics.csv)：全部 TensorBoard scalar，长表格式
- [`raw/velocity_tracking.csv`](raw/velocity_tracking.csv)：全部 1M–20M checkpoints 的逐时刻速度、回报和终止记录
- [`raw/checkpoint_summary.csv`](raw/checkpoint_summary.csv)：每个 checkpoint、每档命令的 MAE、RMSE 和重置次数
- [`raw/command_demo.csv`](raw/command_demo.csv)：GIF 对应的逐时刻命令和实际状态
- [`raw/tensorboard/`](raw/tensorboard/)：原始 TensorBoard event 文件
- [`raw/checkpoints/pupper_ppo_{best_steps}_steps.zip`](raw/checkpoints/pupper_ppo_{best_steps}_steps.zip)：仓库归档的最佳 checkpoint；其余中间 checkpoint 保留在本机训练输出中
- [`raw/machine.json`](raw/machine.json)：机器和软件环境
- [`raw/run_config.yaml`](raw/run_config.yaml)：本次实际运行配置

## 局限与下一步

1. 策略观测没有机身线速度，只能从关节状态间接推断速度，增加了速度闭环控制难度。
2. `ent_coef=0.01` 使策略标准差持续增大；下一轮应尝试 `0.001` 或线性衰减。
3. 低速超调说明单一指数速度奖励不足，可提高 tracking 权重或加入低速/静止专门约束。
4. 当前模型只在平地仿真验证，没有做域随机化和真机部署。
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
    write_report(machine, config, scalars, summary_rows, aggregate, best_steps)
    print(f"报告已生成：{REPORT_DIR / 'README.md'}")
    print(f"最佳 checkpoint：{best_checkpoint}")


if __name__ == "__main__":
    main()
