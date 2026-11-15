"""Bounded running-gait search; score flight, actual speed and original checks."""
import argparse,copy,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scenario_matrix import run_case


def score(r):
    # Passing all original checks dominates every failed candidate.
    if r['passed']:return 1000+r['average_forward_speed_m_s']
    return (20*sum(r['checks'].values())/len(r['checks'])
        +4*min(1,r['average_forward_speed_m_s']/.1)
        +2*min(3,len(r['completed_flight_intervals_s']))/3
        -2*max(0,r['max_driven_joint_speed_rad_s']-10)
        -3*max(0,r['max_heading_deviation_deg']-25)/25
        -5*(not r['checks']['no_body_contact']))


def trial(job):
    label,config,output=job
    return dict(label=label,result=run_case('flat','run_attempt',output=output/label,config_override=config))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/other_tasks/run_refinement'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('reference/run-bound-candidate.json').read_text())
    rng=np.random.default_rng(20261110)
    # frequency, duty, front/rear amplitudes, front/rear biases, slew, heading gain, rear phase.
    mean=np.array([2.2,.6,*[base['amplitude_rad'][i] for i in [0,2]],
        *[base['bias_rad'][i] for i in [0,2]],8.,base['heading_kp'],np.pi])
    sigma=np.array([.25,.06,.05,.05,.06,.06,.7,.15,.25])
    lo=np.array([1.6,.4,.25,.3,-.6,-.4,5.,.1,2.5]);hi=np.array([3.,.7,.6,.65,-.2,0.,8.,.8,3.8])
    records=[];best=None
    with ProcessPoolExecutor(2) as pool:
        for generation in range(3):
            xs=[mean.copy()]+[np.clip(rng.normal(mean,sigma),lo,hi) for _ in range(7)]
            jobs=[]
            for i,x in enumerate(xs):
                cfg=copy.deepcopy(base)
                cfg.update(name='bound_search_candidate',frequency_hz=float(x[0]),sweep_fraction=float(x[1]),
                    amplitude_rad=[x[2],x[2],x[3],x[3]],bias_rad=[x[4],x[4],x[5],x[5]],
                    max_target_speed_rad_s=float(x[6]),heading_kp=float(x[7]),phase_rad=[0.,0.,x[8],x[8]],
                    note='Parameter search candidate; acceptance is determined by metrics, not this label.')
                jobs.append((f'g{generation}_{i}',cfg,a.output))
            rs=list(pool.map(trial,jobs));order=sorted(range(8),key=lambda i:score(rs[i]['result']),reverse=True)
            elites=np.array([xs[i] for i in order[:3]]);mean=elites.mean(axis=0);sigma=np.maximum(elites.std(axis=0),sigma*.5)
            records.extend(rs)
            best=max(records,key=lambda x:score(x['result']))
            (a.output/'summary.json').write_text(json.dumps({'seed':20261110,'algorithm':'CEM','results':records,'best':best},indent=2)+'\n')
            (a.output/'selected.json').write_text(json.dumps(best['result']['controller_config'],indent=2)+'\n')
            r=best['result'];print(generation,best['label'],r['passed'],r['average_forward_speed_m_s'],r['max_driven_joint_speed_rad_s'],len(r['completed_flight_intervals_s']),flush=True)
            if r['passed']:break

if __name__=='__main__':main()
