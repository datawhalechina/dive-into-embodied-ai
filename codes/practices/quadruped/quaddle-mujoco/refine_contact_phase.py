"""Constrain slew and heading on the least-bad contact pilot; bounded follow-up."""
import copy,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case,profile
from search_contact_phase import rank
from curriculum_rl import curriculum_scene


def trial(spec):
    label,config,scene,definition,x=spec
    return dict(label=label,result=run_case(scene,'walk',config_override=config,scene_definition=definition,
        initial_x_offset=x,output=Path('outputs/contact_phase_20261110/refinement')/label))

if __name__=='__main__':
    root=Path('outputs/contact_phase_20261110/refinement');root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('outputs/contact_phase_20261110/search/selected.json').read_text())
    jobs=[]
    for hz in [1.5,1.8]:
        for slew in [5.,6.5]:
            for kp in [.4,.7]:
                jobs.append((f'hz{hz}_slew{slew}_kp{kp}',dict(cfg,frequency_hz=hz,max_target_speed_rad_s=slew,heading_kp=kp),'stairs_up',None,0.))
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for x in pool.map(trial,jobs):
            results.append(x);r=x['result'];print(x['label'],r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
            (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda x:rank(x['result']))
    cfg=best['result']['controller_config']
    (root/'selected.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cases=[('offset_minus','stairs_up',None,-.005),('offset_plus','stairs_up',None,.005),
        ('flat','flat',None,0.),('stairs_1mm','stairs_up',curriculum_scene('stairs_up',1),0.),
        ('stairs_2mm','stairs_up',curriculum_scene('stairs_up',2),0.),
        ('single_4mm','stairs_up',curriculum_scene('stairs_up',3,single_step=True),0.)]
    with ProcessPoolExecutor(2) as pool:
        checks=list(pool.map(trial,[(label,cfg,scene,definition,x) for label,scene,definition,x in cases]))
    (root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    for x in checks:
        r=x['result'];print('VALIDATE',x['label'],r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
    # Gate for a useful training starting point, not a full-task acceptance rule.
    rs=[best['result']]+[x['result'] for x in checks if x['label'].startswith('offset')]
    safe=['upright','no_body_contact','joint_speed','heading']
    gate=all(all(r['checks'][k] for k in safe) and r['forward_m']>=.3 for r in rs)
    (root/'training_gate.json').write_text(json.dumps(dict(eligible=gate,criterion='all three original-terrain starts: upright, no body contact, joint speed and heading pass, forward >= .3m',best_label=best['label']),indent=2)+'\n')
    print('TRAINING START GATE',gate,flush=True)
