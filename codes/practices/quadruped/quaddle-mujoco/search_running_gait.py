"""Compare explicit trot/bound phase families under unchanged running checks."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy,json,math
from scenario_matrix import profile,run_case


def trial(job):
    label,config=job
    result=run_case('flat','run_attempt',output=Path('outputs/terrain_ppo/run_phases')/label,config_override=config)
    return {'label':label,'result':result}

if __name__=='__main__':
    jobs=[]
    for name,phases in [('trot',[0,math.pi,0,math.pi]),('bound',[0,0,math.pi,math.pi]),('pace',[0,math.pi,math.pi,0])]:
        for hz in [2.2,2.8]:
            for duty in [.45,.6]:
                config=copy.deepcopy(profile('run_attempt'))
                config.update(phase_rad=phases,frequency_hz=hz,sweep_fraction=duty)
                jobs.append((f'{name}_{hz}_{duty}',config))
    results=[]
    out=Path('outputs/terrain_ppo/run_phases');out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(trial,jobs):
            results.append(r)
            print(r['label'],r['result']['forward_m'],len(r['result']['completed_flight_intervals_s']),flush=True)
            (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
