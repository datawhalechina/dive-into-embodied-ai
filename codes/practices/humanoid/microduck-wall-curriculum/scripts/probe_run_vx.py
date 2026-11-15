import argparse, torch
from dataclasses import asdict
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--task", default="Mjlab-RouladeDiveWall-Flat-MicroDuck")
a = ap.parse_args()
configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.num_envs
env = ManagerBasedRlEnv(cfg=cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(a.task).clip_actions)
rl_cfg = load_rl_cfg(a.task)
runner = (load_runner_cls(a.task) or OnPolicyRunner)(env, asdict(rl_cfg), device=device)
runner.load(a.checkpoint, map_location=device)
policy = runner.get_inference_policy(device=device)
base = env.env if hasattr(env, "env") else env
robot = base.scene["robot"]
obs, _ = env.reset()
torch.manual_seed(0)
maxs = torch.zeros(a.num_envs)
sums = torch.zeros(a.num_envs); n = 0
for i in range(a.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, _, _ = env.step(act)
    rpos = robot.data.root_link_pose_w
    qw, qz = rpos[:, 3], rpos[:, 6]
    yaw = torch.atan2(2*qw*qz, 1-2*qz*qz)
    fwd = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    v = robot.data.root_link_vel_w[:, :2]
    vx = (v*fwd).sum(1).cpu()
    maxs = torch.maximum(maxs, vx); sums += vx; n += 1
print("env  max_vx  mean_vx")
for i in range(a.num_envs):
    print(f"{i:3d}  {maxs[i]:+.3f}  {sums[i]/n:+.3f}")
