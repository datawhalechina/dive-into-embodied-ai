"""Quaddle 速度跟踪 PPO 训练（Stable-Baselines3，CPU）。

用法（在 codes/practices/quadruped/cs123 下）：
    uv run python 8.rl_quaddle/train.py                          # 默认 2M 步
    uv run python 8.rl_quaddle/train.py --timesteps 200000 --n-envs 8
    uv run python 8.rl_quaddle/train.py --tensorboard            # 需 tensorboard
    uv run python 8.rl_quaddle/train.py --resume 8.rl_quaddle/outputs/ppo_quaddle.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=str(LAB_DIR / "outputs" / "ppo_quaddle.zip"))
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--resume", default=None, help="断点 zip 路径")
    p.add_argument("--perturb", action="store_true", default=True)
    p.add_argument("--no-perturb", dest="perturb", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # `uv run python 8.rl_quaddle/train.py` 时脚本目录自动在 sys.path[0]
    from quaddle_env import QuaddleEnv

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    def make_env(rank: int):
        def _init():
            env = Monitor(QuaddleEnv(perturb_enabled=args.perturb))
            env.reset(seed=args.seed + rank)
            return env

        return _init

    if args.n_envs > 1:
        env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.n_envs)]))
    else:
        env = VecMonitor(DummyVecEnv([make_env(0)]))

    if args.resume and Path(args.resume).exists():
        model = PPO.load(args.resume, env=env, device=args.device)
        print(f"续训自 {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=1.0,
            device=args.device,
            seed=args.seed,
            tensorboard_log=str(LAB_DIR / "outputs" / "tb") if args.tensorboard else None,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -0.5},
            verbose=1,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,  # 每个 env 的步数，SB3 按n_rollout_steps 对齐
        save_path=str(out_path.parent / "ckpt"),
        name_prefix="ppo_quaddle",
        save_vecnormalize=False,
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt_cb)
    finally:
        model.save(str(out_path))
        print(f"模型已保存：{out_path}")
        env.close()


if __name__ == "__main__":
    main()
