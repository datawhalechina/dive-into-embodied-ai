"""SB3 PPO 训练入口，超参对齐 brax-PPO 配方（lr / γ / λ / clip / n_epochs / batch_size）。

CPU + 16 envs SubprocVecEnv 在 ~5-7k env-steps/sec 量级（physics 0.004 / 5 substeps）。

**关于 env-step 预算**：MJX 版本用 8192 个并行 env 跑 149M 步达到峰值；本 CPU 版
只有 16 个 env，单次 rollout 数据多样性少一个量级，梯度更新更频繁但每次方差更
大。同样的 env-step 总数不能直接换算为同样的策略质量——经验上 16-env PPO 通常
要 2-5× 的 env-steps 才能匹敌 8k-env PPO 的最终性能。

实操建议：先把 `--total-steps 150_000_000` 当作起点（约 8 小时 CPU），训完看
`ep_len_mean` 是否仍在涨；如果仍在涨，再续 50-150M。要稳定追上 MJX 149M 峰
值，预计要 300-700M env-steps，10-30 小时不等。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import train_ppo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 6: PPO 训练 Pupper（与 MJX 配方对齐）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=150_000_000,
                        help="env-step 总数。150M 是起点（约 8 小时 CPU）；如要逼近 MJX 8192-env "
                             "在 149M 步的策略质量，本 16-env CPU 配置常常需要 2-5× 更多步数")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="SubprocVecEnv 并行 env 数；CPU 上 16 是个 sweet spot")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="path to checkpoint to warm-start from")
    args = parser.parse_args()

    print(
        f"开始训练: seed={args.seed}, total_steps={args.total_steps:,}, "
        f"n_envs={args.n_envs}, warm_start={args.warm_start}"
    )
    t0 = time.time()
    path = train_ppo(
        seed=args.seed,
        total_timesteps=args.total_steps,
        n_envs=args.n_envs,
        warm_start=args.warm_start,
    )
    wall = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"训练完成: {wall / 60:.1f} min, checkpoint = {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
