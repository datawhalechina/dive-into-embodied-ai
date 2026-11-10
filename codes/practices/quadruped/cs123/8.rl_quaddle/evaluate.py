"""评估 Quaddle PPO 策略：速度跟踪精度与存活率。

用法（在 codes/practices/quadruped/cs123 下）：
    uv run python 8.rl_quaddle/evaluate.py --model 8.rl_quaddle/outputs/ppo_quaddle.zip
    uv run python 8.rl_quaddle/evaluate.py --episodes 20 --perturb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(LAB_DIR / "outputs" / "ppo_quaddle.zip"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--perturb", action="store_true", help="评估时也开启扰动（默认关）")
    args = ap.parse_args()

    from quaddle_env import CMD_VX_RANGE, CMD_VY_RANGE, CMD_WZ_RANGE, QuaddleEnv

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    env = QuaddleEnv(perturb_enabled=args.perturb, max_steps=250)

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        cmd_log, vel_log = [], []
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total += rew
            s = env._body_states()
            cmd_log.append(info["cmd"].copy())
            vel_log.append(np.array([s["lin_vel"][0], s["lin_vel"][1], s["ang_vel"][2]]))
        cmd = np.array(cmd_log)
        vel = np.array(vel_log)
        vx_err = float(np.abs(cmd[:, 0] - vel[:, 0]).mean())
        wz_err = float(np.abs(cmd[:, 2] - vel[:, 2]).mean())
        speed = float(np.abs(vel[:, 0]).mean())
        status = "fall" if terminated else "done"
        rows.append((ep, total, vx_err, wz_err, speed, status))
        print(f"ep{ep:02d}  reward={total:8.1f}  |vx_err|={vx_err*1000:6.1f}mm/s  "
              f"|wz_err|={wz_err:5.2f}rad/s  |vx|={speed*1000:5.1f}mm/s  {status}")

    alive = sum(1 for r in rows if r[5] == "done") / len(rows)
    vx_err = np.mean([r[2] for r in rows]) * 1000
    wz_err = np.mean([r[3] for r in rows])
    speed = np.mean([r[4] for r in rows]) * 1000
    print("-" * 60)
    print(f"存活率 {alive*100:.0f}%   平均|vx误差|={vx_err:.1f}mm/s   "
          f"平均|wz误差|={wz_err:.2f}rad/s   平均|vx|={speed:.1f}mm/s")
    print(f"命令范围 vx={CMD_VX_RANGE} vy={CMD_VY_RANGE} wz={CMD_WZ_RANGE}")
    print("W1 开环基线对照：对称正弦 ~5mm/s 漂移，无方向控制")
    env.close()


if __name__ == "__main__":
    main()
