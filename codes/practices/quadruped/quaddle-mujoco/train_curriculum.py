"""Bounded curriculum pilot, periodic full-physics evaluation and paired snapshots."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from curriculum_rl import CurriculumEnv,curriculum_scene
from mixed_curriculum import MixedCurriculumEnv
from scenario_matrix import run_case

ROOT=Path(__file__).resolve().parent


def evaluate_job(job):
    scene,level,seed,x,policy,output=job
    return run_case(scene,'walk',output=output,policy_path=policy,initial_x_offset=x,
        scene_definition=curriculum_scene(scene,level,seed))


def evaluate(directory,scene,level,gate=False):
    count=20 if gate else 4
    offsets=np.linspace(-.005,.005,count) if gate else [-.005,0.,.005,.0025]
    jobs=[(scene,level,100+i if gate and scene=='rough' else 7,float(x),directory,
        directory/('gate' if gate else 'evaluation')/str(i)) for i,x in enumerate(offsets)]
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
        results=list(pool.map(evaluate_job,jobs))
    summary={'level':level,'passed':sum(r['passed'] for r in results),'trials':count,'results':results}
    (directory/('gate.json' if gate else 'evaluation.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['rough','stairs_up'],default='rough')
    p.add_argument('--gamma',type=float,default=.999)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--chunk',type=int,default=16384)
    p.add_argument('--chunks-per-level',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mixed',action='store_true',help='Replay easier levels and sample four training terrain seeds')
    p.add_argument('--resume',type=Path,help='Continue a compatible version-2 checkpoint at its recorded level')
    a=p.parse_args()
    if not 0<a.gamma<1 or a.chunk<2048 or a.chunks_per_level<1:p.error('Invalid training budget or gamma')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    source_names=['curriculum_rl.py','train_curriculum.py','terrain_rl.py','scenario_matrix.py','scenarios.py','gait_controller.py','gait.json','models/scene.xml']
    if a.mixed: source_names.append('mixed_curriculum.py')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}
    snapshots=a.output/'source';snapshots.mkdir(exist_ok=True)
    for n in source_names:
        target=snapshots/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/n).read_bytes())
    model=None;env=None;last=a.resume;history=[];best_score=None;best=None;started=time.time()
    start_level=1
    if a.resume:
        prior=json.loads((a.resume/'metadata.json').read_text())
        if prior['environment_version']!=2 or prior['gamma']!=a.gamma:
            raise ValueError('Resume requires matching environment version and gamma')
        start_level=prior['level']
    try:
        for level in range(start_level,4):
            raw=make_vec_env(MixedCurriculumEnv if a.mixed else CurriculumEnv,n_envs=4,seed=a.seed,
                env_kwargs={'scene':a.scene,'level':level,'terrain_seed':7},vec_env_cls=SubprocVecEnv)
            new_env=VecNormalize.load(last/'vecnormalize.pkl',raw) if last else VecNormalize(raw,norm_obs=True,norm_reward=False)
            if env is not None:env.close()
            env=new_env
            if model is None and a.resume:
                model=PPO.load(a.resume/'policy.zip',env=env,device='cpu')
                model.set_random_seed(a.seed)
            elif model is None:
                model=PPO('MlpPolicy',env,device='cpu',seed=a.seed,verbose=1,n_steps=512,batch_size=256,
                    n_epochs=5,learning_rate=3e-4,gamma=a.gamma,gae_lambda=.95,ent_coef=.001,
                    policy_kwargs={'net_arch':[64,64],'log_std_init':-1.5})
            else:model.set_env(env)
            advanced=False
            for iteration in range(a.chunks_per_level):
                model.learn(a.chunk,reset_num_timesteps=False)
                last=a.output/f'level{level}_steps{model.num_timesteps}'
                last.mkdir(exist_ok=True)
                model.save(last/'policy');env.save(last/'vecnormalize.pkl')
                np.savez(last/'obs_stats.npz',mean=env.obs_rms.mean,var=env.obs_rms.var)
                meta={'environment_version':2,'observation_dim':60,'residual_scale_rad':.25,'scene':a.scene,
                    'level':level,'gamma':a.gamma,'seed':a.seed,'total_steps':model.num_timesteps,
                    'sampling':'four terrain seeds plus 20% easier levels' if a.mixed else 'fixed terrain seed 7',
                    'resume':str(a.resume) if a.resume else None,
                    'wall_seconds':time.time()-started,'source_sha256':hashes,'hardware_tested':False}
                (last/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
                result=evaluate(last,a.scene,level)
                # Selection first prioritizes all-check passes, then safety, then distance.
                safety=sum(all(r['checks'][k] for k in ['upright','no_body_contact','joint_speed','heading']) for r in result['results'])
                score=(level,result['passed'],safety,float(np.mean([r['forward_m'] for r in result['results']])))
                if best_score is None or score>best_score:best_score=score;best=str(last)
                event={'checkpoint':str(last),'level':level,'steps':model.num_timesteps,'passed':result['passed'],'trials':4,'selection_score':score}
                if result['passed']==4:
                    gate=evaluate(last,a.scene,level,gate=True)
                    event['gate_passed']=gate['passed'];advanced=gate['passed']>=16
                history.append(event)
                report={'scene':a.scene,'gamma':a.gamma,'seed':a.seed,'best_checkpoint':best,'history':history,
                    'wall_seconds':time.time()-started,'completed_level':level if advanced else level-1,
                    'status':'finished' if advanced and level==3 else ('advance' if advanced else 'training')}
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                print('EVALUATION',json.dumps(event),flush=True)
                if advanced:break
            if not advanced:
                report['status']='pilot_budget_exhausted_without_gate_pass'
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n');break
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
