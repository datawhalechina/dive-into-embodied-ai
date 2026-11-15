"""Check serialized 60-D policy rollout against the Gym interface."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from curriculum_rl import CurriculumEnv,CurriculumPolicyController,curriculum_scene
from scenarios import load_scene
from scenario_matrix import profile

p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args()
torch.set_num_threads(1)
meta=json.loads((a.checkpoint/'metadata.json').read_text())
scene=curriculum_scene(meta['scene'],meta['level'])
env=CurriculumEnv(meta['scene'],level=meta['level'],seed_noise=False)
env.config.update(meta.get('controller_config', {}))
obs,_=env.reset(seed=10000)
model,data,rotation=load_scene(scene)
controller=CurriculumPolicyController(model,scene,rotation,profile('walk'),a.checkpoint)
for step in range(2000):
    controller(data);mujoco.mj_step(model,data)
    if step%5==0:mujoco.mj_forward(model,data)
policy=PPO.load(a.checkpoint/'policy.zip',device='cpu')
with np.load(a.checkpoint/'obs_stats.npz') as stats: mean,var=stats['mean'],stats['var']
for tick in range(100):
    normalized=np.clip((obs-mean)/np.sqrt(var+1e-8),-10,10)
    action=policy.predict(normalized,deterministic=True)[0]
    obs,_,done,_,_=env.step(action)
    for step in range(2000+tick*10,2010+tick*10):
        controller(data);mujoco.mj_step(model,data)
        if step%5==0:mujoco.mj_forward(model,data)
    np.testing.assert_allclose(env.data.qpos,data.qpos,rtol=0,atol=1e-12)
    if done:break
print('serialized policy / normalization / observation-history rollout parity PASS',tick+1)
