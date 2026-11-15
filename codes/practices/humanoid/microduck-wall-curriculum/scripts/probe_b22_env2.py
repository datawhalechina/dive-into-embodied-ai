"""B22 5cm 全面探针: 8 env x 多回合, 统计出生姿态/墙距/翻滚完成点/过墙成功率。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import math
import torch
from dataclasses import asdict
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
CKPT = "logs/rsl_rl/microduck_roulade_dive/2026-10-30_23-33-22_microduck_roulade_divewall_run15/model_228468.pt"
N_ENVS, EP_STEPS = 8, 400

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = N_ENVS
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or OnPolicyRunner
runner = runner_cls(wrapper, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
wall_ent = env.scene["wall"]
print("wall entity:", wall_ent is not None)

def pitch_deg(q):
    w, x, y, z = (q[4], q[1], q[2], q[3]) if len(q) >= 7 else (q[0], q[1], q[2], q[3])
    return math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))

stats = []
obs, _ = wrapper.reset()
with torch.inference_mode():
    for ep_step in range(EP_STEPS):
        acts = policy(obs)
        obs, _, dones, extras = wrapper.step(acts)
        if ep_step == 0 or (dones.any() if torch.is_tensor(dones) else dones):
            p = robot.data.root_link_pos_w.detach().cpu().numpy()
            q = robot.data.root_link_pose_w.detach().cpu().numpy()
            wp = wall_ent.data.root_link_pose_w.detach().cpu().numpy() if wall_ent else None
            for i in range(N_ENVS):
                d = math.dist(p[i][:2], wp[i][:2]) if wp is not None else -1
                stats.append((ep_step, i, p[i][0], p[i][1], p[i][2], pitch_deg(q[i]), d))
        # 重置后重新采样出生状态
        if torch.is_tensor(dones) and dones.any():
            p = robot.data.root_link_pos_w.detach().cpu().numpy()
            q = robot.data.root_link_pose_w.detach().cpu().numpy()
            wp = wall_ent.data.root_link_pose_w.detach().cpu().numpy() if wall_ent else None
            for i in torch.nonzero(dones).flatten().tolist():
                d = math.dist(p[i][:2], wp[i][:2]) if wp is not None else -1
                stats.append((ep_step, i, p[i][0], p[i][1], p[i][2], pitch_deg(q[i]), d, "RESET"))

print("step env    x      y      z     pitch   dist2wall")
for row in stats[:60]:
    print(" ".join(f"{v:+.3f}" if isinstance(v, float) else str(v) for v in row))

# 每回合终点: 是否越过墙 (沿出生朝向前向)
print("=== final positions after rollout ===")
p = robot.data.root_link_pos_w.detach().cpu().numpy()
wp = wall_ent.data.root_link_pose_w.detach().cpu().numpy() if wall_ent else None
for i in range(N_ENVS):
    d = math.dist(p[i][:2], wp[i][:2]) if wp is not None else -1
    q = robot.data.root_link_pose_w[i].detach().cpu().numpy()
    side = ""
    if wp is not None:
        v = p[i][:2] - wp[i][:2]
        side = "PAST" if (v @ v / max(1e-9, v @ v) ** .5) > 0 and abs(v[0]) + abs(v[1]) > 0.02 and (v[0] > 0 if abs(v[0]) > abs(v[1]) else v[1] > 0) else ""
    print(f"env{i}: duck=({p[i][0]:+.3f},{p[i][1]:+.3f},{p[i][2]:+.3f}) wall=({wp[i][0]:+.3f},{wp[i][1]:+.3f}) dist={d:.3f} pitch={pitch_deg(q):+.1f} {side}")
print("done")
