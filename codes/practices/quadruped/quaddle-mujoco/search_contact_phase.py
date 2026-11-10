"""Factorial trajectory/contact pilot with original 4mm stair acceptance."""
import copy,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import profile,run_case


def rank(result):
    safety=['upright','no_body_contact','joint_speed','heading']
    return (int(result['passed']),int(all(result['checks'][k] for k in safety)),
        sum(result['checks'].values()),min(2.,result['forward_m']))


def trial(spec):
    name,cfg=spec
    r=run_case('stairs_up','walk',output=Path('outputs/contact_phase_20261110/search')/name,config_override=cfg)
    return dict(label=name,result=r)


def candidates():
    base=profile('walk');jobs=[('original',copy.deepcopy(base))]
    for shape in ['cosine','quintic']:
        cfg=dict(base,contact_phase_controller=True,contact_adaptation=False,trajectory_shape=shape)
        jobs.append((shape+'_open_loop',cfg))
    jobs.append(('landing_only',dict(base,contact_phase_controller=True,contact_adaptation=True,trajectory_shape='quintic',riser_hold_s=0.,riser_angle_rad=0.)))
    for shape in ['cosine','quintic']:
        for angle in [-.15,.1,.25]:
            for rate in [.3,1.2]:
                cfg=dict(base,contact_phase_controller=True,contact_adaptation=True,trajectory_shape=shape,
                    riser_angle_rad=angle,riser_phase_rate=rate,landing_slew_rad_s=3.,landing_hold_s=.04,riser_hold_s=.08)
                jobs.append((f'{shape}_angle{angle}_rate{rate}',cfg))
    return jobs

if __name__=='__main__':
    out=Path('outputs/contact_phase_20261110/search');out.mkdir(parents=True,exist_ok=True)
    snapshots=out/'source';snapshots.mkdir(exist_ok=True)
    hashes={}
    for name in ['contact_phase_controller.py','search_contact_phase.py','scenario_matrix.py','gait_controller.py','gait.json','scenarios.py']:
        data=Path(name).read_bytes();(snapshots/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for x in pool.map(trial,candidates()):
            results.append(x);best=max(results,key=lambda z:rank(z['result']))
            (out/'summary.json').write_text(json.dumps(dict(results=results,best=best,source_sha256=hashes),indent=2)+'\n')
            (out/'selected.json').write_text(json.dumps(best['result']['controller_config'],indent=2)+'\n')
            r=x['result'];print(x['label'],r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
