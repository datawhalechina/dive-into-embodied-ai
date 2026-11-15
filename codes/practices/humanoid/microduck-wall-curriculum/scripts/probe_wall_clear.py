"""Probe: deterministic rollout, check if duck passes wall plane. argv: ckpt sink"""
import os, sys, math
import torch

CKPT, SINK = sys.argv[1], sys.argv[2]
os.environ.update(dict(
    WANDB_MODE="offline", DIVEWALL_STAGE="4", DIVEWALL_WALL_SINK=SINK,
    DIVEWALL_WALL_DIST="0.40,0.65", DIVEWALL_PHASE_B3="1",
))
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import asdict
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK)
runner = runner_cls(venv, asdict(agent_cfg), device=device)
runner.load(CKPT, map_location=device)
pol = runner.get_inference_policy(device=device)

obs, _ = venv.reset()
scene = env.scene
max_past, peak_z = 0.0, 0.0
start_rel = None
with torch.inference_mode():
    for t in range(900):
        obs, _, _, _ = venv.step(pol(obs))
        rpos = scene["robot"].data.root_link_pose_w[0].cpu()
        wpos = scene["wall"].data.root_link_pos_w[0].cpu()
        p = scene["robot"].data.root_link_pos_w[0].cpu()
        yaw = float(torch.atan2(2*rpos[3]*rpos[6], 1-rpos[6]**2))
        fwd = torch.tensor([math.cos(yaw), math.sin(yaw)])
        rel = float((wpos[:2] - p[:2]) @ fwd)
        if start_rel is None:
            start_rel = rel
        max_past = max(max_past, -rel)
        peak_z = max(peak_z, float(p[2]))
print(f"[probe] sink={SINK} start_rel={start_rel:+.2f} max_past_wall={max_past:+.3f} peak_z={peak_z:.3f}")
print("[probe] CLEARED" if max_past > 0.05 else "[probe] NOT cleared")
