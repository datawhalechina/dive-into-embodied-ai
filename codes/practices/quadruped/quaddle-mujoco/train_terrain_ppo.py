"""Reproducible CPU residual PPO stages with paired normalization checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from terrain_rl import TerrainEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='stairs_up')
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.steps < 1 or not 1 <= args.envs <= 8:
        p.error('steps must be positive and envs between 1 and 8')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    env=make_vec_env(TerrainEnv,n_envs=args.envs,seed=args.seed,
        env_kwargs={'scene':args.scene,'seed_noise':True},vec_env_cls=SubprocVecEnv)
    if args.resume:
        env=VecNormalize.load(args.resume/'vecnormalize.pkl',env)
        env.training=True
        model=PPO.load(args.resume/'policy.zip',env=env,device='cpu')
    else:
        env=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10.)
        model=PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=1,n_steps=512,
            batch_size=256,n_epochs=5,learning_rate=3e-4,gamma=.995,gae_lambda=.95,
            ent_coef=.001,policy_kwargs={'net_arch':[64,64],'log_std_init':-1.5})
    source={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest() for name in
        ['terrain_rl.py','train_terrain_ppo.py','gait_controller.py','gait.json','scenarios.py','scenario_matrix.py','models/scene.xml']}
    started=time.time()
    try:
        model.learn(total_timesteps=args.steps,reset_num_timesteps=not bool(args.resume))
        model.save(args.output/'policy')
        env.save(args.output/'vecnormalize.pkl')
        np.savez(args.output/'obs_stats.npz',mean=env.obs_rms.mean,var=env.obs_rms.var)
        result={'scene':args.scene,'seed':args.seed,'requested_stage_steps':args.steps,
            'total_steps':model.num_timesteps,'wall_seconds':time.time()-started,'device':'cpu',
            'residual_scale_rad':.25,'observation_dim':36,'resume':str(args.resume) if args.resume else None,
            'randomization':'initial lateral position +/-5mm only; fixed terrain',
            'source_sha256':source,'versions':{n:importlib.metadata.version(n) for n in ['torch','stable-baselines3','gymnasium','mujoco','numpy']}}
        (args.output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally: env.close()

if __name__=='__main__': main()
