"""Bounded fine-tuning with controller-only ablations and unchanged target tests."""
import argparse,hashlib,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from practice_rl import PracticeEnv,practice_scene
from scenario_matrix import run_case,profile


def evaluate_job(spec):
    scene,checkpoint,scale,x,output=spec
    return run_case(scene,'walk',policy_path=checkpoint,initial_x_offset=x,output=output,
        scene_definition=practice_scene(scene,scale))


def evaluate(scene,checkpoint,ablation=False):
    cases=[(1.,0.),(.375,0.)] if ablation else [(1.,-.005),(1.,0.),(1.,.005),(.375,0.)]
    jobs=[(scene,checkpoint,scale,x,checkpoint/'evaluation'/str(i)) for i,(scale,x) in enumerate(cases)]
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:rs=list(pool.map(evaluate_job,jobs))
    result=[dict(scale=scale,offset=x,result=r) for (scale,x),r in zip(cases,rs)]
    (checkpoint/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EVAL',checkpoint,[(x['scale'],x['result']['passed'],round(x['result']['forward_m'],3)) for x in result],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['rough','stairs_up'],required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    if a.output.exists():raise ValueError('Use a new output directory to preserve evidence')
    prior=json.loads((a.resume/'metadata.json').read_text())
    if prior['environment_version']!=2 or prior['scene']!=a.scene:raise ValueError('Matching 60D terrain checkpoint required')
    a.output.mkdir(parents=True)
    protocol=dict(terrain_scales=[.25,.375,.5,1.],probabilities=[.3,.3,.3,.1],terrain_seeds=[7,11,13,17],
        initial_yaw_deg=[-3,3],friction_scale=[.9,1.1],seed=43,learning_rate=1e-4,new_steps_per_task=32768,
        resume_curriculum_level=prior.get('level'))
    (a.output/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=['practice_rl.py','train_practice.py','curriculum_rl.py','terrain_rl.py','scenario_matrix.py','scenarios.py','gait_controller.py','gait.json','models/scene.xml']
    hashes={}
    for n in names:
        data=Path(n).read_bytes();target=a.output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[n]=hashlib.sha256(data).hexdigest()
    import shutil
    config=dict(profile('walk'),speed_feedback_gain=.02,speed_feedback_threshold=8.)
    baseline=[]
    for label,gain in [('prior_policy',0.),('feedback_only',.02)]:
        d=a.output/label;d.mkdir()
        for n in ['policy.zip','obs_stats.npz','vecnormalize.pkl']:shutil.copy2(a.resume/n,d/n)
        meta=dict(prior,controller_config=dict(config,speed_feedback_gain=gain),experiment_variant=label,
            source_sha256=hashes,resume=str(a.resume),new_training_steps=0)
        (d/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        baseline.append(dict(label=label,evaluations=evaluate(a.scene,d,True)))
    raw=make_vec_env(PracticeEnv,n_envs=4,seed=43,env_kwargs={'scene':a.scene},vec_env_cls=SubprocVecEnv)
    env=VecNormalize.load(a.resume/'vecnormalize.pkl',raw)
    model=PPO.load(a.resume/'policy.zip',env=env,device='cpu',learning_rate=1e-4)
    model.set_random_seed(43);initial_steps=model.num_timesteps;start=time.time();history=[]
    try:
        for iteration in range(2):
            model.learn(16384,reset_num_timesteps=False)
            d=a.output/f'new_steps{model.num_timesteps-initial_steps}';d.mkdir()
            model.save(d/'policy');env.save(d/'vecnormalize.pkl');np.savez(d/'obs_stats.npz',mean=env.obs_rms.mean,var=env.obs_rms.var)
            meta=dict(prior,controller_config=config,experiment_variant='practice_finetune',training_environment='PracticeEnv-v1',
                sampling='PracticeEnv: scales .25/.375/.5/1 with probabilities .3/.3/.3/.1; yaw +/-3deg; sliding friction .9..1.1',
                seed=43,new_training_steps=model.num_timesteps-initial_steps,total_steps=model.num_timesteps,
                source_sha256=hashes,resume=str(a.resume),wall_seconds=time.time()-start)
            (d/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            history.append(dict(checkpoint=str(d),evaluations=evaluate(a.scene,d)))
            report=dict(scene=a.scene,baseline=baseline,history=history,new_training_steps=model.num_timesteps-initial_steps,
                wall_seconds=time.time()-start,status='finished' if iteration==1 else 'training',hardware_tested=False)
            (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()

if __name__=='__main__':main()
