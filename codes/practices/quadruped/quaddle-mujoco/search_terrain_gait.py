"""Bounded evolutionary gait training; validation uses unchanged task checks."""
import argparse,copy,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scenario_matrix import run_case,profile


def trial(job):
    name,mode,config,out=job
    return run_case(name,mode,output=out,config_override=config)


def score(r):
    c=r['checks']
    # A successful rollout dominates a failed one; reward speed only within constraints.
    return (30*sum(c.values())/len(c)+10*int(r['passed'])+min(2,r['forward_m'])
        -5*max(0.,r['max_driven_joint_speed_rad_s']-10)
        -3*max(0.,r['max_heading_deviation_deg']-25)/25
        -10*int(not c['no_body_contact']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--mode',default='walk')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path)
    p.add_argument('--generations',type=int,default=3);a=p.parse_args()
    rng=np.random.default_rng(20261109)
    # Hz, feedback, steering limit, target slew, front/rear amplitudes, front/rear biases.
    base=json.loads(a.base.read_text()) if a.base else profile(a.mode)
    mean=np.array([base['frequency_hz'],base['heading_kp'],base['steering_limit'],base['max_target_speed_rad_s'],
        base['amplitude_rad'][0],base['amplitude_rad'][2],base['bias_rad'][0],base['bias_rad'][2]]) if a.base else np.array([2.2,.45,.3,6.,.405,.481,-.421,-.222])
    sigma=np.array([.35,.2,.12,1.,.08,.08,.09,.09])
    lo=np.array([1.2,.1,.1,4.,.2,.25,-.65,-.45]);hi=np.array([3.2,1.,.6,8.,.65,.7,-.1,.05])
    records=[];best=None
    a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(2) as pool:
        for generation in range(a.generations):
            xs=[mean]+[np.clip(rng.normal(mean,sigma),lo,hi) for _ in range(7)]
            jobs=[]
            for i,x in enumerate(xs):
                cfg=copy.deepcopy(base)
                cfg.update(frequency_hz=float(x[0]),heading_kp=float(x[1]),steering_limit=float(x[2]),
                    max_target_speed_rad_s=float(x[3]),amplitude_rad=[*([float(x[4])]*2),*([float(x[5])]*2)],
                    bias_rad=[*([float(x[6])]*2),*([float(x[7])]*2)])
                jobs.append((a.scene,a.mode,cfg,a.output/f'g{generation}_{i}'))
            rs=list(pool.map(trial,jobs))
            order=sorted(range(len(rs)),key=lambda i:score(rs[i]),reverse=True)
            elites=np.array([xs[i] for i in order[:3]])
            mean=elites.mean(axis=0);sigma=np.maximum(elites.std(axis=0),sigma*.5)
            for r in rs:
                records.append(r)
                if best is None or score(r)>score(best):best=r
            (a.output/'search.json').write_text(json.dumps({'seed':20261109,'algorithm':'CEM','results':records,'best':best},indent=2)+'\n')
            (a.output/'selected.json').write_text(json.dumps(best['controller_config'],indent=2)+'\n')
            print(generation,'best',best['forward_m'],[k for k,v in best['checks'].items() if not v],flush=True)
            if best['passed']:break

if __name__=='__main__':main()
