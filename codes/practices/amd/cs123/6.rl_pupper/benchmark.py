"""用固定命令和种子量化 Pupper checkpoint 的存活与速度跟踪表现。"""

from __future__ import annotations

import argparse
import json
from math import acos, exp, sqrt
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pupper_env import PupperEnv
from train import require_rocm_device


SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "lateral": (0.0, 0.3, 0.0),
    "yaw": (0.0, 0.0, 0.8),
}
DEFAULT_SEEDS = (11, 22, 33)


def run_episode(model, env, scenario, command, seed, seconds):
    obs, _ = env.reset(seed=seed)
    env.cmd = np.asarray(command, dtype=np.float32)
    max_eval_steps = round(seconds / env.dt)
    velocities = []
    tilts = []
    rewards = []
    terminated = truncated = False

    for _ in range(max_eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rotation = env.data.xmat[env._base_id].reshape(3, 3)
        local_lin = rotation.T @ env.data.qvel[0:3]
        local_ang = rotation.T @ env.data.qvel[3:6]
        velocities.append((local_lin[0], local_lin[1], local_ang[2]))
        tilts.append(acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
        rewards.append(reward)
        if terminated or truncated:
            break

    velocity = np.asarray(velocities, dtype=np.float64)
    target = np.asarray(command, dtype=np.float64)
    error = velocity - target
    survival_s = len(rewards) * env.dt
    return {
        "scenario": scenario,
        "seed": seed,
        "command": list(command),
        "survival_s": survival_s,
        "survival_ratio": survival_s / seconds,
        "completed": len(rewards) == max_eval_steps,
        "episode_return": float(sum(rewards)),
        "vx_mean": float(np.mean(velocity[:, 0])),
        "vy_mean": float(np.mean(velocity[:, 1])),
        "wz_mean": float(np.mean(velocity[:, 2])),
        "linear_rmse": float(sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
        "yaw_rmse": float(sqrt(np.mean(error[:, 2] ** 2))),
        "tilt_deg_mean": float(np.degrees(np.mean(tilts))),
        "terminated": terminated,
        "truncated": truncated,
    }


def _mean(episodes, key):
    return float(np.mean([episode[key] for episode in episodes]))


def summarize(episodes):
    by_scenario = {}
    for scenario in SCENARIOS:
        selected = [episode for episode in episodes if episode["scenario"] == scenario]
        by_scenario[scenario] = {
            "survival_ratio": _mean(selected, "survival_ratio"),
            "completion_rate": _mean(selected, "completed"),
            "episode_return": _mean(selected, "episode_return"),
            "linear_rmse": _mean(selected, "linear_rmse"),
            "yaw_rmse": _mean(selected, "yaw_rmse"),
            "tilt_deg_mean": _mean(selected, "tilt_deg_mean"),
        }

    survival = _mean(episodes, "survival_ratio")
    linear_rmse = _mean(episodes, "linear_rmse")
    yaw_rmse = _mean(episodes, "yaw_rmse")
    tilt_deg = _mean(episodes, "tilt_deg_mean")
    translation_episodes = [
        episode
        for episode in episodes
        if episode["scenario"] in {"forward", "backward", "lateral"}
    ]
    yaw_episodes = [
        episode for episode in episodes if episode["scenario"] == "yaw"
    ]
    translation_rmse = _mean(translation_episodes, "linear_rmse")
    yaw_command_rmse = _mean(yaw_episodes, "yaw_rmse")
    score = 100.0 * (
        0.40 * survival
        + 0.30 * exp(-translation_rmse / 0.25)
        + 0.20 * exp(-yaw_command_rmse / 0.5)
        + 0.10 * exp(-tilt_deg / 15.0)
    )
    return {
        "score": score,
        "survival_ratio": survival,
        "completion_rate": _mean(episodes, "completed"),
        "episode_return": _mean(episodes, "episode_return"),
        "linear_rmse": linear_rmse,
        "yaw_rmse": yaw_rmse,
        "translation_rmse": translation_rmse,
        "yaw_command_rmse": yaw_command_rmse,
        "tilt_deg_mean": tilt_deg,
        "by_scenario": by_scenario,
    }


def benchmark(checkpoint, seconds, seeds):
    device = require_rocm_device()
    env = PupperEnv(max_steps=max(round(seconds / 0.02) + 1, 1000))
    model = PPO.load(checkpoint, env=env, device=device)
    episodes = [
        run_episode(model, env, scenario, command, seed, seconds)
        for scenario, command in SCENARIOS.items()
        for seed in seeds
    ]
    env.close()
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "seconds_per_episode": seconds,
        "seeds": list(seeds),
        "summary": summarize(episodes),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="量化 Pupper PPO checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = benchmark(args.checkpoint, args.seconds, tuple(args.seeds))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"评估结果已保存：{args.output}")
    summary = result["summary"]
    print(
        f"score={summary['score']:.2f} "
        f"survival={summary['survival_ratio']:.1%} "
        f"linear_rmse={summary['linear_rmse']:.3f}m/s "
        f"yaw_rmse={summary['yaw_rmse']:.3f}rad/s "
        f"tilt={summary['tilt_deg_mean']:.1f}deg",
    )


if __name__ == "__main__":
    main()
