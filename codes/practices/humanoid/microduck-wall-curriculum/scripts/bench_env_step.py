# Bench: 拆解 collection 时间 — 纯 env.step vs 策略前向 vs rsl_rl rollout 完整路径
import os, time
for k, v in {
    "WANDB_MODE": "offline",
    "DIVEWALL_STAGE": "4", "DIVEWALL_WALL_SINK": "0.070", "DIVEWALL_WALL_DIST": "0.40,0.65",
    "DIVEWALL_POSTDIVE_PROB": "0.0", "DIVEWALL_PHASE_B1": "0", "DIVEWALL_PHASE_B3": "1",
    "DIVEWALL_PHASE_B17": "0", "DIVEWALL_PHASE_B18": "1", "DIVEWALL_B18_MIDROLL_FRAC": "0.22",
    "DIVEWALL_PHASE_B21": "1", "MUJOCO_GL": "egl",
}.items():
    os.environ[k] = v

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
N_ENVS = 2048
env_cfg = load_env_cfg(TASK)
env_cfg.scene.num_envs = N_ENVS
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env.reset()

class Actor(torch.nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.net = torch.nn.Sequential(
            torch.nn.Linear(i, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, o))
    def act(s, o):
        return s.net(o)

_sp = env.observation_space
obs_shape = (_sp.spaces["actor"].shape if hasattr(_sp,"spaces") else _sp["actor"].shape)
ac = Actor(obs_shape[0], env.action_space.shape[1]).cuda()

def sync(): torch.cuda.synchronize()

# --- 1) 纯 env.step（物理+obs+reward），随机动作 ---
actions = torch.randn(N_ENVS, env.action_space.shape[1], device="cuda") * 0.1
for _ in range(5): obs, rew, term, trunc, extras = env.step(actions)  # warmup/compile
sync()
t0 = time.perf_counter(); N = 100
for _ in range(N):
    obs, rew, term, trunc, extras = env.step(actions)
sync()
t_step = (time.perf_counter() - t0) / N
print(f"[env.step]     {t_step*1000:8.2f} ms/step  -> {N_ENVS/t_step:,.0f} steps/s")

# --- 2) 纯策略前向 ---
obs_t = obs["actor"] if hasattr(obs, "keys") else obs
for _ in range(5): ac.act(obs_t)
sync()
t0 = time.perf_counter()
for _ in range(240): ac.act(obs_t)
sync()
t_fwd = (time.perf_counter() - t0) / 240
print(f"[actor fwd]    {t_fwd*1000:8.2f} ms/fwd")

# --- 3) 串行完整路径（近似 rsl_rl collection 一个 step）---
for _ in range(5):
    a = ac.act(obs_t); obs, rew, term, trunc, extras = env.step(a)
sync()
t0 = time.perf_counter(); N = 100
for _ in range(N):
    a = ac.act(obs_t)
    obs, rew, term, trunc, extras = env.step(a)
sync()
t_full = (time.perf_counter() - t0) / N
print(f"[fwd+step]     {t_full*1000:8.2f} ms/step")
print(f"[对比] rsl_rl 实测 collection 1.22s/24steps = {1220/24:.1f} ms/step")
env.close()
