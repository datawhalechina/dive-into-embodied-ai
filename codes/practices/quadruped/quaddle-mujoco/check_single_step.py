"""Single-riser feasibility search under original matrix physical thresholds."""
from concurrent.futures import ProcessPoolExecutor
import copy,json
from pathlib import Path
from curriculum_rl import curriculum_scene
from scenario_matrix import profile,run_case


def trial(job):
    level,label,config=job
    result=run_case('stairs_up','walk',output=Path('outputs/curriculum/single_step')/f'level{level}'/label,
        scene_definition=curriculum_scene('stairs_up',level,single_step=True),config_override=config)
    return dict(level=level,label=label,result=result)

if __name__=='__main__':
    configs={'original':profile('walk')}
    for name in ['downhill3','uphill5']:
        configs[name]=json.loads((Path('reference')/f'terrain-{name}.json').read_text())
    for hz,front,rear in [(1.5,-.32,-.22),(1.8,-.32,-.12),(2.2,-.32,-.22),(1.5,-.52,-.12),(2.,-.22,-.32)]:
        config=copy.deepcopy(profile('walk'));config.update(frequency_hz=hz,bias_rad=[front,front,rear,rear])
        configs[f'hz{hz}_front{front}_rear{rear}']=config
    out=Path('outputs/curriculum/single_step');out.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(trial,[(level,name,cfg) for level in [1,2,3] for name,cfg in configs.items()]):
            results.append(r)
            (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            x=r['result'];print(r['level'],r['label'],x['passed'],round(x['forward_m'],3),[k for k,v in x['checks'].items() if not v],flush=True)
