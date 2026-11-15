"""Bounded support/swing scheduling experiment, retaining original task checks."""
import json, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from search_contact_phase import rank
from curriculum_rl import curriculum_scene

ROOT=Path('outputs/support_coordination_20261110')

def trial(spec):
    label,cfg,scene,definition,offset=spec
    result=run_case(scene,'walk',config_override=cfg,scene_definition=definition,
                    initial_x_offset=offset,output=ROOT/label)
    print(label,result['passed'],round(result['forward_m'],3),
          [k for k,v in result['checks'].items() if not v],flush=True)
    return dict(label=label,result=result)

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('outputs/contact_phase_20261110/refinement/selected.json').read_text())
    jobs=[('previous_contact',cfg,'stairs_up',None,0.)]
    for gate in [.65,.75,.85]:
        for feet in [1,2]:
            for rate in [.5,1.]:
                new=dict(cfg,support_coordination=True,contact_adaptation=False,
                    swing_gate_phase=gate,minimum_support_feet=feet,support_phase_rate=rate,
                    maximum_swing_feet=1)
                jobs.append((f'gate{gate}_feet{feet}_rate{rate}',new,'stairs_up',None,0.))
    source=ROOT/'source';source.mkdir(exist_ok=True);hashes={}
    for name in ['support_coordinator.py','contact_phase_controller.py','search_support_coordination.py','scenario_matrix.py','gait_controller.py','gait.json','scenarios.py']:
        b=Path(name).read_bytes();(source/name).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
    (ROOT/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(2) as pool:
        for x in pool.map(trial,jobs):
            rows.append(x);(ROOT/'search.json').write_text(json.dumps(rows,indent=2)+'\n')
    best=max(rows[1:],key=lambda x:rank(x['result']));cfg=best['result']['controller_config']
    (ROOT/'selected.json').write_text(json.dumps(best,indent=2)+'\n')
    specs=[('offset_minus','stairs_up',None,-.005),('offset_plus','stairs_up',None,.005),
           ('flat','flat',None,0.),('single_4mm','stairs_up',curriculum_scene('stairs_up',3,single_step=True),0.),
           ('stairs_2mm','stairs_up',curriculum_scene('stairs_up',2),0.)]
    with ProcessPoolExecutor(2) as pool:
        rows=list(pool.map(trial,[(label,cfg,scene,definition,x) for label,scene,definition,x in specs]))
    (ROOT/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
