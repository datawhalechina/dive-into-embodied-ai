"""生成 gait 条件迁移微调的报告、曲线、GIF 与全部原始数据。"""

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

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


BASE_REPORT_DIR = Path(__file__).resolve().parent
LAB_DIR = BASE_REPORT_DIR.parent
REPORT_DIR = BASE_REPORT_DIR / "gait_finetune"
RAW_DIR = REPORT_DIR / "raw"
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
TB_DIR = RAW_DIR / "tensorboard"
RUN_DIR = LAB_DIR / "outputs" / "gait_finetune_5m"
CONFIG_PATH = LAB_DIR / "configs" / "finetune_gait.yaml"

sys.path.insert(0, str(LAB_DIR))
from pupper_env import GAIT_NAMES, PupperEnv  # noqa: E402


COMMAND_VX = 0.4
EVAL_SECONDS = 8.0
WARMUP_SECONDS = 1.0
GIF_FPS = 12
FOOT_LABELS = ("FR", "FL", "RR", "RL")


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
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return info


def copy_training_artifacts() -> tuple[dict, Path]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    shutil.copy2(CONFIG_PATH, RAW_DIR / "run_config.yaml")
    events = sorted((RUN_DIR / "tb").rglob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"TensorBoard event 不存在：{RUN_DIR / 'tb'}")
    TB_DIR.mkdir(parents=True, exist_ok=True)
    for event in events:
        shutil.copy2(event, TB_DIR / event.name)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(RUN_DIR.glob("pupper_gait_finetune_*_steps.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"训练 checkpoints 不存在：{RUN_DIR}")
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, CHECKPOINT_DIR / checkpoint.name)
    shutil.copy2(
        RUN_DIR / "pupper_gait_finetune_final.zip",
        CHECKPOINT_DIR / "pupper_gait_finetune_final.zip",
    )
    return config, events[0]


def export_training_metrics(event_path: Path) -> dict[str, list]:
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    scalars = {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get("scalars", [])
    }
    with (RAW_DIR / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("tag", "step", "wall_time", "value"))
        for tag, events in sorted(scalars.items()):
            for event in events:
                writer.writerow((
                    tag, event.step, f"{event.wall_time:.6f}", f"{event.value:.10g}",
                ))
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
        "Episode reward", "Episode length", "Explained variance",
        "Policy std", "Clip fraction", "Approx KL",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, tag, title in zip(axes.flat, tags, titles):
        events = scalars.get(tag, [])
        axis.plot(
            [event.step / 1e6 for event in events],
            [event.value for event in events],
            linewidth=1.2,
        )
        axis.set_title(title)
        axis.set_xlabel("Fine-tuning steps [M]")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Gait-conditioned PPO transfer fine-tuning")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(REPORT_DIR / "training_curves.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def checkpoint_steps(path: Path) -> int:
    return int(path.stem.split("_")[-2])


def evaluate_checkpoints() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("pupper_gait_finetune_*_steps.zip"),
        key=checkpoint_steps,
    )
    env = PupperEnv(gait_enabled=True, gait_switch_steps=0)

    for checkpoint in checkpoints:
        steps = checkpoint_steps(checkpoint)
        model = PPO.load(checkpoint, device="cpu")
        for gait_name in GAIT_NAMES:
            obs, _ = env.reset(seed=42)
            env.set_gait(gait_name)
            env.cmd = np.array([COMMAND_VX, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()
            resets = 0
            gait_rows = []
            for index in range(int(EVAL_SECONDS / env.dt)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rotation = env.data.xmat[env._base_id].reshape(3, 3)
                local_linear = rotation.T @ env.data.qvel[0:3]
                row = {
                    "checkpoint_steps": steps,
                    "gait": gait_name,
                    "time_s": index * env.dt,
                    "phase": info["gait_phase"],
                    "command_vx": COMMAND_VX,
                    "actual_vx": float(local_linear[0]),
                    "contact_match": info["gait_contact_match"],
                    "reward": float(reward),
                    "base_z": float(env.data.qpos[2]),
                    "terminated": int(terminated),
                    "episode": resets,
                }
                for foot_index, foot in enumerate(FOOT_LABELS):
                    row[f"expected_{foot.lower()}"] = info["expected_contacts"][foot_index]
                    row[f"actual_{foot.lower()}"] = info["actual_contacts"][foot_index]
                rows.append(row)
                gait_rows.append(row)
                if terminated or truncated:
                    resets += 1
                    obs, _ = env.reset(seed=42 + resets)
                    env.set_gait(gait_name)
                    env.cmd = np.array([COMMAND_VX, 0.0, 0.0], dtype=np.float32)
                    obs = env._get_obs()

            measured = [row for row in gait_rows if row["time_s"] >= WARMUP_SECONDS]
            errors = np.array([row["actual_vx"] - COMMAND_VX for row in measured])
            summaries.append({
                "checkpoint_steps": steps,
                "gait": gait_name,
                "mean_actual_vx": float(np.mean([row["actual_vx"] for row in measured])),
                "velocity_mae": float(np.mean(np.abs(errors))),
                "contact_match": float(np.mean([row["contact_match"] for row in measured])),
                "resets": resets,
            })

    with (RAW_DIR / "gait_evaluation.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (RAW_DIR / "checkpoint_summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(summaries[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    return rows, summaries


def aggregate_checkpoints(summaries: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in summaries:
        grouped[int(row["checkpoint_steps"])].append(row)
    aggregate = []
    for steps, rows in sorted(grouped.items()):
        velocity_mae = float(np.mean([row["velocity_mae"] for row in rows]))
        contact_match = float(np.mean([row["contact_match"] for row in rows]))
        resets = int(sum(row["resets"] for row in rows))
        aggregate.append({
            "checkpoint_steps": steps,
            "velocity_mae": velocity_mae,
            "contact_match": contact_match,
            "resets": resets,
            "score": velocity_mae + 0.25 * (1.0 - contact_match) + 0.25 * resets,
        })
    with (RAW_DIR / "checkpoint_aggregate.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(aggregate[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate)
    return aggregate


def plot_checkpoint_comparison(aggregate: list[dict]) -> None:
    steps = np.array([row["checkpoint_steps"] for row in aggregate]) / 1e6
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(steps, [row["velocity_mae"] for row in aggregate], marker="o")
    axes[0].set_ylabel("Velocity MAE [m/s]")
    axes[1].plot(steps, [row["contact_match"] for row in aggregate], marker="o")
    axes[1].set_ylabel("Contact schedule match")
    axes[1].set_ylim(0.0, 1.02)
    axes[2].plot(steps, [row["resets"] for row in aggregate], marker="o")
    axes[2].set_ylabel("Resets across 3 gaits")
    for axis in axes:
        axis.set_xlabel("Fine-tuning steps [M]")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Gait checkpoint evaluation at vx=0.4 m/s")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(REPORT_DIR / "checkpoint_comparison.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_contact_patterns(rows: list[dict], best_steps: int) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for row_index, gait_name in enumerate(GAIT_NAMES):
        selected = [
            row for row in rows
            if row["checkpoint_steps"] == best_steps
            and row["gait"] == gait_name
            and 1.0 <= row["time_s"] < 3.0
        ]
        for column, prefix in enumerate(("expected", "actual")):
            matrix = np.array([
                [row[f"{prefix}_{foot.lower()}"] for row in selected]
                for foot in FOOT_LABELS
            ])
            axes[row_index, column].imshow(
                matrix, aspect="auto", interpolation="nearest", cmap="Blues",
                vmin=0, vmax=1, origin="lower", extent=(1.0, 3.0, -0.5, 3.5),
            )
            axes[row_index, column].set_ylim(3.5, -0.5)
            axes[row_index, column].set_yticks(range(4), FOOT_LABELS)
            axes[row_index, column].set_ylabel(gait_name)
            axes[row_index, column].grid(False)
    axes[0, 0].set_title("Desired contact schedule")
    axes[0, 1].set_title("Measured foot contacts")
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    figure.suptitle(f"Best gait checkpoint contact patterns — {best_steps / 1e6:g}M")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(REPORT_DIR / "contact_patterns.webp", dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_gait_demo(model: PPO) -> None:
    env = PupperEnv(gait_enabled=True, gait_switch_steps=0)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.distance = 1.4
    camera.elevation = -20.0
    camera.azimuth = 90.0
    frames = []
    raw_rows = []
    next_frame_time = 0.0
    global_time = 0.0
    obs, _ = env.reset(seed=42)
    try:
        for gait_name in GAIT_NAMES:
            env.set_gait(gait_name)
            env.cmd = np.array([COMMAND_VX, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()
            for _ in range(int(3.0 / env.dt)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rotation = env.data.xmat[env._base_id].reshape(3, 3)
                actual_vx = float((rotation.T @ env.data.qvel[0:3])[0])
                raw_rows.append({
                    "time_s": global_time,
                    "gait": gait_name,
                    "phase": info["gait_phase"],
                    "command_vx": COMMAND_VX,
                    "actual_vx": actual_vx,
                    "contact_match": info["gait_contact_match"],
                    "reward": reward,
                    "terminated": int(terminated),
                })
                if global_time + 0.5 * env.dt >= next_frame_time:
                    camera.lookat[:] = env.data.xpos[env._base_id]
                    renderer.update_scene(env.data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((10, 10, 280, 58), fill=(0, 0, 0))
                    draw.text(
                        (20, 20),
                        f"gait={gait_name}  vx_cmd={COMMAND_VX:.1f} m/s",
                        fill=(255, 255, 255),
                    )
                    frames.append(np.asarray(frame))
                    next_frame_time += 1.0 / GIF_FPS
                global_time += env.dt
                if terminated or truncated:
                    obs, _ = env.reset(seed=42)
                    env.set_gait(gait_name)
                    env.cmd = np.array([COMMAND_VX, 0.0, 0.0], dtype=np.float32)
                    obs = env._get_obs()
    finally:
        renderer.close()

    imageio.mimsave(
        REPORT_DIR / "gait_demo.gif", frames, duration=1000 / GIF_FPS, loop=0,
    )
    with (RAW_DIR / "gait_demo.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(raw_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(raw_rows)


def _scalar_stats(scalars: dict[str, list], tag: str) -> tuple[float, float, float]:
    values = [event.value for event in scalars[tag]]
    return values[0], values[-1], max(values)


def write_report(
    machine: dict,
    config: dict,
    scalars: dict[str, list],
    summaries: list[dict],
    aggregate: list[dict],
    best_steps: int,
) -> None:
    first_reward, final_reward, peak_reward = _scalar_stats(
        scalars, "rollout/ep_rew_mean",
    )
    first_length, final_length, peak_length = _scalar_stats(
        scalars, "rollout/ep_len_mean",
    )
    all_events = [event for events in scalars.values() for event in events]
    final_step = max(event.step for event in all_events)
    duration_minutes = (
        max(event.wall_time for event in all_events)
        - min(event.wall_time for event in all_events)
    ) / 60
    best_rows = [row for row in summaries if row["checkpoint_steps"] == best_steps]
    best_aggregate = next(
        row for row in aggregate if row["checkpoint_steps"] == best_steps
    )
    table = "\n".join(
        f"| {row['gait']} | {row['mean_actual_vx']:.3f} | "
        f"{row['velocity_mae']:.3f} | {row['contact_match']:.3f} | "
        f"{row['resets']} |"
        for row in best_rows
    )
    gpu_line = next(
        (line.strip() for line in machine["gpu"].splitlines() if "Chipset Model" in line),
        "Chipset Model: unknown",
    )
    report = f"""# Pupper gait 条件迁移训练报告

## 任务目标

实验一只给策略目标速度 `(vx, vy, wz)`。速度决定“机身怎么移动”，却不能唯一决定“四只脚按什么顺序接触地面”：同样以 0.4 m/s 前进，可以采用 walk、trot 或 pace。因此实验二把控制目标扩展为：

> 在跟踪 `(vx, vy, wz)` 的同时，按指定 `gait_id` 和周期相位执行对应的足端接触时序。

策略输入由 45 维扩为 50 维。原 45 维顺序不变，末尾追加 `walk/trot/pace` 三维 one-hot 与 `sin(phase)、cos(phase)`。动作仍是 12 个关节位置残差。新增奖励比较 FR、FL、RR、RL 的实际接触与目标接触序列，`gait_id` 因而不仅改变输入，也改变每个时刻的正确行为。

## 实验性质

这是从实验一 **19M 步 checkpoint 迁移后的 5M 步微调**，用于尽快验证 gait 条件链路，不是带 gait 与不带 gait 的公平消融实验。迁移时复制旧策略的全部兼容参数，两个首层网络的前 45 列原样保留，新 5 列置零后学习。

公平对照已另存为 `configs/train_no_gait.yaml` 与 `configs/train_gait.yaml`：二者随机种子、20M 训练预算和 PPO 超参数一致，并都从零训练。

## 结论

Apple M3 Ultra CPU 使用 16 个并行 MuJoCo 环境完成 {final_step:,} 步微调，TensorBoard 记录跨度约 {duration_minutes:.1f} 分钟。按速度 MAE、接触匹配率和摔倒次数的组合分数，最佳 checkpoint 为 **[{best_steps / 1e6:g}M](raw/checkpoints/pupper_gait_finetune_{best_steps}_steps.zip)**。

最佳 checkpoint 在三种 gait 上的平均速度 MAE 为 {best_aggregate['velocity_mae']:.3f} m/s，接触时序逐足匹配率为 {best_aggregate['contact_match']:.3f}，共重置 {best_aggregate['resets']} 次。速度与稳定性得到保留，但 0.5 左右的接触匹配接近“未按相位区分触地”的基准水平；接触图中的实际触地频率也明显高于目标周期。因此本次迁移**没有成功形成标准 walk、trot 和 pace**，只能证明 gait 输入、相位、奖励、迁移和评估链路已经跑通。

## 训练机器与参数

- 芯片：{gpu_line.split(':', 1)[-1].strip()}
- CPU 逻辑核心：{machine['logical_cpu_count']}
- 统一内存：{machine['memory_bytes'] / 1024 ** 3:.0f} GiB
- 训练设备：CPU，16 个 `SubprocVecEnv`
- 微调步数：{config['run']['total_timesteps']:,}
- 学习率：{config['ppo']['learning_rate']}
- 熵系数：{config['ppo']['ent_coef']}
- 裁剪范围：{config['ppo']['clip_range']}
- gait 接触奖励权重：{config['gait']['contact_reward_weight']}
- gait 切换间隔：{config['gait']['switch_steps']} 控制步，即 10 秒

完整参数见 [`raw/run_config.yaml`](raw/run_config.yaml)。

## 训练曲线

![训练曲线](training_curves.webp)

- `ep_rew_mean`：{first_reward:.2f} → {final_reward:.2f}，峰值 {peak_reward:.2f}
- `ep_len_mean`：{first_length:.0f} → {final_length:.0f}，峰值 {peak_length:.0f}

![checkpoint 对比](checkpoint_comparison.webp)

## 最佳策略评估

三种 gait 均固定 `vx=0.4 m/s`，每种运行 8 秒，去掉首秒热身：

| gait | 平均 vx | 速度 MAE | 接触匹配率 | 重置 |
| --- | ---: | ---: | ---: | ---: |
{table}

![目标与实际接触](contact_patterns.webp)

![gait 演示](gait_demo.gif)

GIF 依次输入 walk、trot、pace，每段 3 秒，速度命令始终为 0.4 m/s，以隔离 gait 条件。当前视觉差异和接触时序都不足以把三段认定为标准步态。

## 原始数据

- [`raw/training_metrics.csv`](raw/training_metrics.csv)：全部 TensorBoard scalar
- [`raw/gait_evaluation.csv`](raw/gait_evaluation.csv)：全部 checkpoint、三种 gait 的逐控制步状态与四足接触
- [`raw/checkpoint_summary.csv`](raw/checkpoint_summary.csv)：每个 checkpoint 和 gait 的汇总指标
- [`raw/checkpoint_aggregate.csv`](raw/checkpoint_aggregate.csv)：checkpoint 选择分数
- [`raw/gait_demo.csv`](raw/gait_demo.csv)：GIF 对应逐时刻数据
- [`raw/tensorboard/`](raw/tensorboard/)：原始 TensorBoard event
- [`raw/checkpoints/pupper_gait_finetune_{best_steps}_steps.zip`](raw/checkpoints/pupper_gait_finetune_{best_steps}_steps.zip)：仓库归档的最佳 checkpoint；其余中间 checkpoint 保留在本机训练输出中
- [`../raw/checkpoints/pupper_ppo_19000000_steps.zip`](../raw/checkpoints/pupper_ppo_19000000_steps.zip)：迁移源模型
- [`raw/machine.json`](raw/machine.json)：机器与软件环境
- [`raw/run_config.yaml`](raw/run_config.yaml)：实际训练配置

## 后续公平对照

不能用本次 5M 迁移结果直接断言 gait 方案优于原模型。正式报告的消融实验应分别执行：

```bash
uv run python 6.rl_pupper/train.py --config 6.rl_pupper/configs/train_no_gait.yaml
uv run python 6.rl_pupper/train.py --config 6.rl_pupper/configs/train_gait.yaml
```

两组都从零训练 20M 步，再以相同速度跟踪、稳定性和接触时序指标评估。`train_gait.yaml` 已根据本次 0.5 权重不足的结果，将接触奖励权重提高到 4.0；PPO 超参数和训练预算仍与无 gait 组一致。
"""
    (REPORT_DIR / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    machine = save_machine_info()
    config, event_path = copy_training_artifacts()
    scalars = export_training_metrics(event_path)
    plot_training_curves(scalars)
    rows, summaries = evaluate_checkpoints()
    aggregate = aggregate_checkpoints(summaries)
    plot_checkpoint_comparison(aggregate)
    best = min(aggregate, key=lambda row: row["score"])
    best_steps = int(best["checkpoint_steps"])
    plot_contact_patterns(rows, best_steps)
    best_path = CHECKPOINT_DIR / f"pupper_gait_finetune_{best_steps}_steps.zip"
    render_gait_demo(PPO.load(best_path, device="cpu"))
    write_report(machine, config, scalars, summaries, aggregate, best_steps)
    print(f"报告已生成：{REPORT_DIR / 'README.md'}")
    print(f"最佳 checkpoint：{best_path}")


if __name__ == "__main__":
    main()
