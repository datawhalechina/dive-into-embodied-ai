# torch profiler: 抓一个完整 iteration 的 kernel timeline (skip 前4个iter, 捕获第5个)
import os, sys
for k, v in {
    "WANDB_MODE": "offline",
    "DIVEWALL_STAGE": "4", "DIVEWALL_WALL_SINK": "0.070", "DIVEWALL_WALL_DIST": "0.40,0.65",
    "DIVEWALL_POSTDIVE_PROB": "0.0", "DIVEWALL_PHASE_B1": "0", "DIVEWALL_PHASE_B3": "1",
    "DIVEWALL_PHASE_B17": "0", "DIVEWALL_PHASE_B18": "1", "DIVEWALL_B18_MIDROLL_FRAC": "0.22",
    "DIVEWALL_PHASE_B21": "1", "MUJOCO_GL": "egl",
}.items():
    os.environ[k] = v

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from dataclasses import asdict
from pathlib import Path

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
env_cfg = load_env_cfg(TASK); env_cfg.scene.num_envs = 2048
rl_cfg = load_rl_cfg(TASK); rl_cfg.max_iterations = 5
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
agent_cfg = asdict(rl_cfg)
runner = MjlabOnPolicyRunner(env, agent_cfg, "logs/profiler_run", "cuda:0")

from torch.profiler import profile, ProfilerActivity, schedule

_orig_log = runner.logger.log
def _log_and_step(*a, **kw):
    r = _orig_log(*a, **kw)
    prof.step()   # advance profiler schedule once per iteration
    return r
runner.logger.log = _log_and_step

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(skip_first=3, wait=1, warmup=0, active=1, repeat=1),
    record_shapes=False, profile_memory=False,
) as prof:
    runner.learn(num_learning_iterations=5, init_at_random_ep_len=True)

out = Path("logs/profiler_out"); out.mkdir(exist_ok=True)
prof.export_chrome_trace(str(out / "one_iter.json"))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
# gap 分析: CUDA 总时间 vs wall
stats = prof.key_averages()
print("\n-- CUDA self time sum vs iteration wall (1.14s) 见上表 --")
env.close()
