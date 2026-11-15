"""B22 5cm 环境探针: 实测墙位置/鸭子出生点/翻滚起点 (play mode, 同 B22 env vars)。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import math
import numpy as np
import torch
from dataclasses import asdict
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
CKPT = "logs/rsl_rl/microduck_roulade_dive/2026-10-30_23-33-22_microduck_roulade_divewall_run15/model_228468.pt"

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or OnPolicyRunner
runner = runner_cls(wrapper, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

robot = env.scene["robot"]

def pitch_deg(q):  # quat (w,x,y,z)
    w, x, y, z = (q[4], q[1], q[2], q[3]) if len(q) >= 7 else (q[0], q[1], q[2], q[3])
    return math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))

obs, _ = wrapper.reset()
p0 = robot.data.root_link_pos_w.detach().cpu().numpy()
print("duck spawn pos per env:")
for i in range(env_cfg.scene.num_envs):
    print(f"  env{i}: ({p0[i][0]:+.3f}, {p0[i][1]:+.3f}, {p0[i][2]:+.3f})")

# 从观测里拿 wall_rel (obs term index 12, shape (3,))
wall_info = {}
with torch.inference_mode():
    for step in range(60):
        acts = policy(obs)
        obs, _, _, _ = wrapper.step(acts)
        if step % 10 == 0:
            p = robot.data.root_link_pos_w[0].detach().cpu().numpy()
            q = robot.data.root_link_pose_w[0].detach().cpu().numpy()
            print(f"step {step:3d}: pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) pitch={pitch_deg(q):+7.1f}")

# 直接查 reset_wall event 的实际参数
ev = env_cfg.events["reset_wall"]
print("reset_wall params:", {k: v for k, v in ev.params.items() if k in ("distance_range", "lateral_range", "sink", "height_range", "wall_height", "sink_range")})
# wall 真实位置: 从 unwrapper 找
try:
    pe = env.unwrapped
    for attr in dir(pe):
        if "wall" in attr.lower():
            print("env attr:", attr, getattr(pe, attr))
except Exception as e:
    print("attr scan fail:", e)
print("done probe")
