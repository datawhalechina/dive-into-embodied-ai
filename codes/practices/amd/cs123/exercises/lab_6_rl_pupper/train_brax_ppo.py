"""brax-PPO + MJX 训练脚本。

用法（单卡 GPU）：
    python train_brax_ppo.py \\
        --num-envs 8192 --num-timesteps 200000000 \\
        --output portfolio/pupper_mjx --tb-dir tb_mjx

依赖：本仓库 `envs/pupper_env_mjx.py`。jax / brax / mujoco-mjx 的锁定版本与
CUDA 12 / 13 / 纯 CPU 三套装法见 `lab_6_rl_pupper/requirements-mjx.txt`，装在
独立 venv（勿与主线 `.venv` 混）。

验证结果：200M 步训练 ep_rew 在约 149M 步处达到峰值 51.15 ± 7.4；存活率
100%，yaw tracking 持续提升。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from lab_6_rl_pupper.envs.pupper_env_mjx import PupperV3MJXEnv  # noqa: E402

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


ENV_REGISTRY = {
    "upstream": ("pupper_v3_mjx", PupperV3MJXEnv),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_REGISTRY.keys()), default="upstream",
                   help="MJX baseline 配方（36-dim obs, 18 reward terms, 域随机化, "
                        "cmd vx±0.75/vy±0.5/wz±2.0）")
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--num-timesteps", type=int, default=200_000_000)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--num-updates-per-batch", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-cost", type=float, default=0.01)
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--clipping-epsilon", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-evals", type=int, default=10)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True, help="保存目录")
    p.add_argument("--tb-dir", type=str, default=None, help="tensorboard 日志目录")
    p.add_argument("--policy-hidden", type=str, default="128,128,128,128",
                   help="逗号分隔的 policy MLP 隐层尺寸")
    p.add_argument("--value-hidden", type=str, default="256,256,256,256,256",
                   help="逗号分隔的 value MLP 隐层尺寸")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    tb_dir = Path(args.tb_dir) if args.tb_dir else out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir)) if SummaryWriter is not None else None

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"num_envs={args.num_envs} num_timesteps={args.num_timesteps:,}", flush=True)
    print(f"output={out_dir}  tb={tb_dir}", flush=True)

    env_name, env_cls = ENV_REGISTRY[args.env]
    envs.register_environment(env_name, env_cls)
    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)
    print(f"env={args.env} ({env_name}) obs_dim={env.observation_size} act_dim={env.action_size}", flush=True)

    policy_hidden = tuple(int(x) for x in args.policy_hidden.split(","))
    value_hidden = tuple(int(x) for x in args.value_hidden.split(","))
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
    )

    start_time = time.time()
    history = []

    def progress_fn(num_steps: int, metrics: dict):
        elapsed = time.time() - start_time
        sps = num_steps / max(elapsed, 1e-6)
        ep_rew = float(metrics.get("eval/episode_reward", float("nan")))
        ep_rew_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        ep_len = float(metrics.get("eval/episode_length", float("nan")))
        line = (
            f"[{int(elapsed):>5}s] step={num_steps:>11,}  "
            f"ep_rew={ep_rew:.3f}±{ep_rew_std:.3f}  ep_len={ep_len:.1f}  "
            f"sps={sps:,.0f}"
        )
        print(line, flush=True)
        record = {"step": int(num_steps), "elapsed": elapsed, **{k: float(v) for k, v in metrics.items() if np.isscalar(v) or hasattr(v, "item")}}
        history.append(record)
        (out_dir / "progress.jsonl").open("a").write(json.dumps(record) + "\n")
        if writer is not None:
            writer.add_scalar("wall/sps", sps, num_steps)
            for k, v in metrics.items():
                try:
                    writer.add_scalar(k, float(v), num_steps)
                except Exception:
                    pass
            writer.flush()

    saved = {"params": None, "step": 0}

    def policy_params_fn(current_step, make_policy, params):
        saved["params"] = params
        saved["step"] = int(current_step)
        ckpt = out_dir / f"params_step_{int(current_step):010d}.pkl"
        with ckpt.open("wb") as fh:
            pickle.dump(params, fh)
        print(f"  saved checkpoint {ckpt.name}", flush=True)

    train_fn = partial(
        ppo.train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=args.reward_scaling,
        episode_length=args.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        discounting=args.discounting,
        learning_rate=args.lr,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        seed=args.seed,
        num_eval_envs=args.num_eval_envs,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )

    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

    final_path = out_dir / "params_final.pkl"
    with final_path.open("wb") as fh:
        pickle.dump(params, fh)
    print(f"final params saved to {final_path}", flush=True)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
