"""使用 Stable-Baselines3 PPO 训练 Pupper 速度跟踪策略。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from pupper_env import (
    REWARD_WEIGHTS,
    TRACKING_ANG_SIGMA,
    TRACKING_LIN_SIGMA,
    PupperEnv,
)


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = LAB_DIR / "outputs"


class TrainingDiagnosticsCallback(BaseCallback):
    """把奖励分量和跌倒率写入 TensorBoard。"""

    def _on_rollout_start(self) -> None:
        self.component_sums = {}
        self.sample_count = 0
        self.terminated_count = 0
        self.truncated_count = 0

    def _on_step(self) -> bool:
        for index, info in enumerate(self.locals["infos"]):
            for key, value in info.items():
                if key.startswith("r_"):
                    self.component_sums[key] = (
                        self.component_sums.get(key, 0.0) + float(value)
                    )
            self.sample_count += 1
            if self.locals["dones"][index]:
                if info.get("TimeLimit.truncated", False):
                    self.truncated_count += 1
                else:
                    self.terminated_count += 1
        return True

    def _on_rollout_end(self) -> None:
        count = max(self.sample_count, 1)
        for key, total in self.component_sums.items():
            self.logger.record(f"reward_components/{key}", total / count)
        finished = self.terminated_count + self.truncated_count
        if finished:
            self.logger.record(
                "episodes/fall_rate",
                self.terminated_count / finished,
            )


def require_rocm_device() -> str:
    """返回 PyTorch 的 ROCm 设备；不可用时拒绝回退到 CPU。"""
    if torch.version.hip is None:
        raise RuntimeError(
            "当前 PyTorch 不含 ROCm/HIP；请在 cs123 目录执行 uv sync。",
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch ROCm 已安装，但未发现可用 AMD GPU；训练不会回退到 CPU。",
        )

    device = "cuda:0"  # PyTorch 在 ROCm 上沿用 torch.cuda 设备 API。
    try:
        probe = torch.ones(1, device=device)
        if (probe * 2).item() != 2.0:
            raise RuntimeError("ROCm 张量计算结果异常")
        torch.cuda.synchronize(0)
    except Exception as exc:
        raise RuntimeError(f"ROCm GPU 自检失败；训练不会回退到 CPU：{exc}") from exc

    print(
        f"使用 ROCm {torch.version.hip}："
        f"{torch.cuda.get_device_name(0)} ({device})",
    )
    return device


def make_env(seed: int, rank: int):
    def _thunk():
        env = Monitor(PupperEnv())
        env.reset(seed=seed + rank)
        return env

    return _thunk


def train(
    timesteps: int,
    n_envs: int,
    seed: int,
    out: str,
    tensorboard: bool,
    checkpoint: str | None,
    learning_rate: float = 1e-4,
    ent_coef: float = 1e-3,
    batch_size: int = 512,
    log_std_init: float = -0.5,
    target_kl: float = 0.02,
) -> Path:
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")

    device = require_rocm_device()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "timesteps": timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "batch_size": batch_size,
        "log_std_init": log_std_init,
        "target_kl": target_kl,
        "checkpoint": checkpoint,
        "reward_weights": REWARD_WEIGHTS,
        "tracking_lin_sigma": TRACKING_LIN_SIGMA,
        "tracking_ang_sigma": TRACKING_ANG_SIGMA,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
    }
    (out_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    vec_env = SubprocVecEnv([make_env(seed, rank) for rank in range(n_envs)])
    try:
        tensorboard_log = str(out_dir / "tb") if tensorboard else None

        if checkpoint:
            print(f"从 checkpoint 继续训练：{checkpoint}")
            model = PPO.load(
                checkpoint,
                env=vec_env,
                tensorboard_log=tensorboard_log,
                device=device,
            )
            model.set_random_seed(seed)
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=2048,
                batch_size=batch_size,
                n_epochs=4,
                learning_rate=learning_rate,
                gamma=0.97,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                target_kl=target_kl,
                policy_kwargs={
                    "net_arch": [256, 256],
                    "log_std_init": log_std_init,
                },
                tensorboard_log=tensorboard_log,
                verbose=1,
                seed=seed,
                device=device,
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1_000_000 // n_envs, 1),
            save_path=str(out_dir),
            name_prefix="pupper_ppo",
        )
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList([
                checkpoint_callback,
                TrainingDiagnosticsCallback(),
            ]),
        )
        final_path = out_dir / "pupper_ppo.zip"
        model.save(str(final_path))
    finally:
        vec_env.close()

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 PPO 训练 Pupper")
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    args = parser.parse_args()

    path = train(
        args.timesteps,
        args.n_envs,
        args.seed,
        args.out,
        args.tensorboard,
        args.checkpoint,
        args.learning_rate,
        args.ent_coef,
        args.batch_size,
        args.log_std_init,
        args.target_kl,
    )
    print(f"模型已保存：{path}")


if __name__ == "__main__":
    main()
