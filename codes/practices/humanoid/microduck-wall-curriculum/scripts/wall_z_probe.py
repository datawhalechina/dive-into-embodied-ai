"""Probe wall placement & dynamics: print wall z at reset and over steps."""
import sys
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

configure_torch_backends()
task = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
env_cfg = load_env_cfg(task)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)


def wall_z():
    wall = env.scene["wall"]
    return float(wall.data.root_link_pos_w[0, 2])


obs, _ = env.reset()
print(f"after reset: wall z = {wall_z():.4f} (expect 0.06-0.113=-0.053)")
zero = torch.zeros((1, env.action_manager.total_action_dim), device="cpu")
for i in range(400):
    obs, r, t, to, info = env.step(zero)
    if i % 40 == 0:
        print(f"step {i}: wall z = {wall_z():.4f}")
env.close()
