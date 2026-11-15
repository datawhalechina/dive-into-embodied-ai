"""Probe: how does the divewall-run2 policy actually get past the wall?

Tracks wall pose (displacement / tilt), robot z, feet contact, and crossing
of the wall plane. 8 envs, 150 steps, deterministic.

Usage: .venv/bin/python scripts/probe_wall_behavior.py \
    --checkpoint logs/rsl_rl/microduck_roulade_dive/<run>/model_57993.pt
"""

import argparse
import math

import torch

import mjlab.tasks  # noqa: F401
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
    rl_cfg = load_rl_cfg(TASK)
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    env_base = env.env if hasattr(env, "env") else env
    robot = env_base.scene["robot"]
    wall = env_base.scene["wall"]
    obs, _ = env.reset()
    cr = env_base.scene.terrain.env_origins
    torch.manual_seed(0)

    # Fixed per-env spawn heading (run-9 lesson: using the CURRENT robot yaw
    # makes "past the wall" flicker true mid-roll when yaw swings — false
    # ground-level "crossings"). Measure progress along the spawn heading.
    rp0 = robot.data.root_link_pose_w.clone()
    yaw0 = torch.atan2(2 * rp0[:, 3] * rp0[:, 6], 1 - 2 * (rp0[:, 5] ** 2 + rp0[:, 6] ** 2))
    fwd0 = torch.stack((torch.cos(yaw0), torch.sin(yaw0)), dim=1)

    # Run-12 fix: root_link_pose_w can be stale right after reset writes
    # (pre-reset default XML pose), inflating wall_dxy by the env-lane
    # spacing. Prefer the recorded spawn pose from reset_wall_ahead.
    wall0 = getattr(env_base, "_wall_spawn_pose", None)
    if wall0 is not None:
        wall0 = wall0.clone()
    else:
        wall0 = wall.data.root_link_pose_w.clone()
    logs = []
    with torch.inference_mode():
        for t in range(args.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            wallp = wall.data.root_link_pose_w
            rz = robot.data.root_link_pos_w[:, 2] - cr[:, 2]
            # wall tilt: quat -> "uprightness" of wall = 1-2(qy^2+qz^2) with yaw-only baseline
            q = wallp[:, 3:7]
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            wdxy = (wallp[:, :2] - wall0[:, :2]).norm(dim=1)
            # robot relative to wall along the FIXED spawn heading
            rp = robot.data.root_link_pose_w
            past = ((rp[:, :2] - wallp[:, :2]) * fwd0).sum(dim=1)
            logs.append((rz.clone(), past.clone(), wdxy.clone(), up.clone()))

    rz = torch.stack([l[0] for l in logs])   # T,N
    past = torch.stack([l[1] for l in logs])
    wdxy = torch.stack([l[2] for l in logs])
    up = torch.stack([l[3] for l in logs])

    print(f"{'env':>3} {'crossed@':>8} {'z_at_cross':>10} {'max_z':>7} {'airborne_z>0.14':>16} "
          f"{'wall_dxy_max':>12} {'wall_up_min':>11}")
    for e in range(args.num_envs):
        crossed = (past[:, e] > 0.05).nonzero()
        c_t = crossed[0].item() if len(crossed) else -1
        z_c = rz[c_t, e].item() if c_t >= 0 else float("nan")
        air = (rz[:, e] > 0.14).sum().item()  # above standing z+~2.5cm
        print(f"{e:>3} {c_t:>8} {z_c:>10.3f} {rz[:, e].max().item():>7.3f} {air:>16} "
              f"{wdxy[:, e].max().item():>12.3f} {up[:, e].min().item():>11.2f}")


if __name__ == "__main__":
    main()
