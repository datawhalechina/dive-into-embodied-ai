"""Per-bucket speed tracking test for a velocity checkpoint.

For each commanded vx bucket (0.2 ... 1.2), fix the twist command, walk 4 s
(200 steps), report mean achieved forward vx over the last 3 s + fall count.

Usage:
  .venv/bin/python scripts/eval_velocity_buckets.py --checkpoint <model.pt>
"""

import argparse
import json
from pathlib import Path

from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-VelocityFast-Flat-MicroDuck"
BUCKETS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
WARMUP = 50
STEPS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--out", default="logs/eval_velocity_buckets.json")
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 12.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    rl_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = (runner_cls or OnPolicyRunner)(wrapper, asdict(rl_cfg), device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    root = env.scene["robot"]

    def set_twist(vx):
        term = env.command_manager.get_term("twist")
        v = torch.tensor([vx, 0.0, 0.0], device=device)
        term.vel_command_b[:] = v
        term.vel_command_w[:] = v

    results = []
    with torch.inference_mode():
        for vx_cmd in BUCKETS:
            obs, _ = wrapper.reset()
            vxs, n_fell = [], 0
            for t in range(STEPS):
                set_twist(vx_cmd)
                obs, _, _, _ = wrapper.step(policy(obs))
                z = root.data.root_link_pos_w[:, 2]
                tilt_ok = z > 0.05
                v_w = torch.nan_to_num(root.data.root_link_lin_vel_w, nan=0.0)
                q = torch.nan_to_num(root.data.root_link_quat_w, nan=0.0)
                fwd = torch.zeros_like(v_w); fwd[:, 0] = 1.0
                def quat_rotate(q, v):
                    w, xyz = q[..., :1], q[..., 1:]
                    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
                    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)
                heading_dir = quat_rotate(q, fwd)
                heading_dir[:, 2] = 0.0
                heading_dir = torch.nn.functional.normalize(heading_dir, dim=-1)
                vx = (v_w * heading_dir).sum(-1) * tilt_ok.float()
                if t >= WARMUP:
                    vxs.append(vx[tilt_ok].cpu())
                    fell = (~tilt_ok).sum().item()
                    n_fell += fell
            all_vx = torch.cat(vxs) if vxs else torch.zeros(1)
            r = {
                "cmd": vx_cmd,
                "n": int(all_vx.numel()),
                "mean_vx": round(float(all_vx.mean()), 3),
                "median_vx": round(float(all_vx.median()), 3),
                "p90_vx": round(float(all_vx.quantile(0.9)), 3),
                "upright_ratio": round(float(all_vx.numel()) / ((STEPS - WARMUP) * args.num_envs), 3),
                "fall_steps": int(n_fell),
            }
            print(json.dumps(r), flush=True)
            results.append(r)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[eval_velocity_buckets] saved -> {args.out}")


if __name__ == "__main__":
    main()
