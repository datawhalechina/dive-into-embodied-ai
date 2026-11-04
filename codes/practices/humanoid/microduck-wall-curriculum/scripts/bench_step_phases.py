# Deep bench: 拆解 env.step 内部各阶段耗时（manager 级插桩 + cuda 同步计时）
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
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

N_ENVS = 2048
env_cfg = load_env_cfg("Mjlab-RouladeDiveWall-Flat-MicroDuck")
env_cfg.scene.num_envs = N_ENVS
print(f"decimation = {env_cfg.decimation}")
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env.reset()

T = {}  # phase -> cumulative seconds

def timed(name, fn):
    def wrapper(*a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn(*a, **kw)
        torch.cuda.synchronize(); T[name] = T.get(name, 0.0) + time.perf_counter() - t0
        return r
    return wrapper

# 插桩
env.action_manager.process_action = timed("1_action_process", env.action_manager.process_action)
env.action_manager.apply_action = timed("2_action_apply(decim)", env.action_manager.apply_action)
env.scene.write_data_to_sim = timed("3_scene_write(decim)", env.scene.write_data_to_sim)
env.sim.step = timed("4_sim.step(physics)", env.sim.step)
env.scene.update = timed("5_scene.update(decim)", env.scene.update)
env.metrics_manager.compute_substep = timed("6_metrics_substep", env.metrics_manager.compute_substep)
env.termination_manager.compute = timed("7_termination", env.termination_manager.compute)
env.reward_manager.compute = timed("8_reward", env.reward_manager.compute)
env.sim.forward = timed("9_sim.forward", env.sim.forward)
env.command_manager.compute = timed("10_command", env.command_manager.compute)
env.event_manager.apply = timed("11_events", env.event_manager.apply)
env.sim.sense = timed("12_sense", env.sim.sense)
env.observation_manager.compute = timed("13_observation", env.observation_manager.compute)

actions = torch.randn(N_ENVS, env.action_space.shape[1], device="cuda") * 0.1
for _ in range(10): env.step(actions)   # warmup + compile
T.clear()
N = 50
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N): env.step(actions)
torch.cuda.synchronize(); total = time.perf_counter() - t0

print(f"\n=== env.step 总计 {(total/N)*1000:.2f} ms/step ({N_ENVS/total:,.0f} steps/s) ===")
acc = 0.0
for k in sorted(T):
    ms = T[k] / N * 1000
    acc += ms
    print(f"{k:28s} {ms:8.2f} ms  ({T[k]/total*100:5.1f}%)")
print(f"{'(sum instrumented)':28s} {acc:8.2f} ms  ({acc/(total/N*1000)*100:5.1f}%)")
env.close()
