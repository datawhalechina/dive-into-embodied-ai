# Bench: rsl_rl rollout 循环逐组件计时 — act / normalizer / storage / logger / env.step
import os, time, torch
for k, v in {
    "WANDB_MODE": "offline",
    "DIVEWALL_STAGE": "4", "DIVEWALL_WALL_SINK": "0.070", "DIVEWALL_WALL_DIST": "0.40,0.65",
    "DIVEWALL_POSTDIVE_PROB": "0.0", "DIVEWALL_PHASE_B1": "0", "DIVEWALL_PHASE_B3": "1",
    "DIVEWALL_PHASE_B17": "0", "DIVEWALL_PHASE_B18": "1", "DIVEWALL_B18_MIDROLL_FRAC": "0.22",
    "DIVEWALL_PHASE_B21": "1", "MUJOCO_GL": "egl",
}.items():
    os.environ[k] = v

from tensordict import TensorDict
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import __name__ as _u  # noqa

N_ENVS = 2048
cfg = load_env_cfg("Mjlab-RouladeDiveWall-Flat-MicroDuck")
cfg.scene.num_envs = 2048
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
venv = RslRlVecEnvWrapper(env)
obs = venv.get_observations()

OBS_A, OBS_C, NACT = 61, 77, env.action_manager.total_action_dim
actor = MLPModel(obs, {"actor": ["actor"]}, "actor", NACT, hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0}).cuda()
critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True).cuda()

T = {}
def timed(name, fn):
    def w(*a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn(*a, **kw)
        torch.cuda.synchronize(); T[name] = T.get(name, 0) + time.perf_counter() - t0
        return r
    return w

def sync(): torch.cuda.synchronize()

# 组件级
for _ in range(10):
    a = actor(obs, stochastic_output=True); v = critic(obs)
    lp = actor.get_output_log_prob(a.detach())
sync()
def comp_bench(name, fn, n=200):
    for _ in range(20): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(n): fn()
    sync(); print(f"{name:32s} {(time.perf_counter()-t0)/n*1000:7.3f} ms")

comp_bench("actor fwd (stoch)", lambda: actor(obs, stochastic_output=True))
comp_bench("critic fwd", lambda: critic(obs))
_a = actor(obs, stochastic_output=True).detach()
comp_bench("log_prob", lambda: actor.get_output_log_prob(_a))
comp_bench("actor.update_normalization", lambda: actor.update_normalization(obs))
comp_bench("critic.update_normalization", lambda: critic.update_normalization(obs))
comp_bench("venv.step (real actions)", lambda: venv.step(_a), n=100)

# 逐步全链 (act→step→process) 计时
with torch.inference_mode():
    for _ in range(10):
        actions = actor(obs, stochastic_output=True)
        obs, rewards, dones, extras = venv.step(actions)
    sync(); N = 100
    tA = tS = tP = 0.0
    for _ in range(N):
        sync(); t0 = time.perf_counter()
        actions = actor(obs, stochastic_output=True)
        sync(); t1 = time.perf_counter()
        obs, rewards, dones, extras = venv.step(actions)
        sync(); t2 = time.perf_counter()
        actor.update_normalization(obs); critic.update_normalization(obs)
        sync(); t3 = time.perf_counter()
        tA += t1-t0; tS += t2-t1; tP += t3-t2
    print(f"\n[rollout chain] act={tA/N*1000:.2f} ms  env.step={tS/N*1000:.2f} ms  normalizers={tP/N*1000:.2f} ms")
env.close()
