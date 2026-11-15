"""Evaluate a saved Dreamer checkpoint on held-out DMC initial states.
Run from experiment root: venv/bin/python evaluate.py --run runs/reacher_easy_seed0
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import av
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import elements
from dreamerv3.main import make_agent,make_env

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--count',type=int,default=5);p.add_argument('--output',type=Path);p.add_argument('--start-index',type=int,default=100);p.add_argument('--modes',nargs='+',choices=['random','dreamer'],default=['random','dreamer']);args=p.parse_args()
 config=elements.Config.load(args.run/'config.yaml')
 agent=make_agent(config);cp=elements.Checkpoint(args.run/'ckpt');cp.agent=agent;cp.step=elements.Counter();checkpoint=args.checkpoint or cp.latest();cp.load(checkpoint,keys=['agent','step'])
 out=args.output or args.run/'heldout-evaluation';out.mkdir(exist_ok=True);results=[]
 (out/'metadata.json').write_text(json.dumps({'training_step':int(cp.step),'checkpoint':str(checkpoint),'task':config.task,'model_training_seed':config.seed}))
 for mode in args.modes:
  for index in range(args.start_index,args.start_index+args.count):
   env=make_env(config,index);rng=np.random.default_rng(index+1000)
   action={'action':np.zeros(2,np.float32),'reset':np.array(True)}
   carry=agent.init_policy(1);rewards=[];states=[];actions=[];frames=[]
   while True:
    obs=env.step(action)
    if not obs['is_first']:rewards.append(float(obs['reward']))
    states.append({k:np.asarray(v).tolist() for k,v in obs.items() if not k.startswith('log/')})
    if index==args.start_index and len(states)%2==1:frames.append(obs['log/image'])
    if obs['is_last']:break
    if mode=='dreamer':
     batched={k:np.asarray(v)[None] for k,v in obs.items() if not k.startswith('log/')}
     carry,acts,_=agent.policy(carry,batched,mode='eval');act=acts['action'][0]
    else:act=rng.uniform(-1,1,2).astype(np.float32)
    actions.append(np.asarray(act).tolist());action={'action':np.asarray(act,np.float32),'reset':np.array(False)}
   env.close();r={'mode':mode,'env_index':index,'environment_seed':hash((config.seed,index))%(2**32-1),'return':sum(rewards),'steps':len(rewards),'target_occupancy':float(np.mean(np.array(rewards)>.5)),'last_100_target_occupancy':float(np.mean(np.array(rewards[-100:])>.5))}
   results.append(r);(out/f'{mode}_{index}.json').write_text(json.dumps({'metrics':r,'observations':states,'actions':actions}))
   if frames:
    with av.open(str(out/f'{mode}_{index}.mp4'),'w') as container:
     stream=container.add_stream('libx264',rate=25);stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
     for frame in frames:
      for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):container.mux(packet)
     for packet in stream.encode():container.mux(packet)
   print(r,flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
