"""Additional single 4 mm step search with explicit phase and stance variants."""
import copy,json,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from curriculum_rl import curriculum_scene
from scenario_matrix import run_case,profile


def trial(job):
    label,config=job
    result=run_case('stairs_up','walk',output=Path('outputs/curriculum/step_phases')/label,
        scene_definition=curriculum_scene('stairs_up',3,single_step=True),config_override=config)
    return dict(label=label,result=result)

if __name__=='__main__':
    jobs=[]
    phases={'crawl':[0,math.pi,math.pi/2,3*math.pi/2],'trot':[0,math.pi,0,math.pi],
        'bound':[0,0,math.pi,math.pi]}
    for name,phase in phases.items():
        for hz in [1.3,1.8]:
            for front,rear in [(-.25,-.4),(-.55,-.1)]:
                cfg=copy.deepcopy(profile('walk'))
                cfg.update(phase_rad=phase,frequency_hz=hz,bias_rad=[front,front,rear,rear],
                    amplitude_rad=[.5,.5,.5,.5],max_target_speed_rad_s=6.)
                jobs.append((f'{name}_{hz}_{front}_{rear}',cfg))
    out=Path('outputs/curriculum/step_phases');out.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(trial,jobs):
            results.append(r);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            x=r['result'];print(r['label'],x['passed'],x['forward_m'],[k for k,v in x['checks'].items() if not v],flush=True)
