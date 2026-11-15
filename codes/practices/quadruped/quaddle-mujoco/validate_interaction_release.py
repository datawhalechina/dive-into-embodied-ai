"""Frozen-config command holdouts; does not tune on the results."""
import copy,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from command_interaction import replay,SCHEDULE
ROOT=Path('outputs/baseline_20261110/interaction_release')
def trial(spec):
    label,kwargs,schedule=spec
    r=replay(ROOT/label,schedule=schedule,**kwargs)
    print(label,r['passed'],[k for k,v in r['physical_checks'].items() if not v],[(s.get('final_1s_max_heading_error_deg'),s['passed']) for s in r['segments']],flush=True)
    return dict(label=label,result=r)
if __name__=='__main__':
    shifted=copy.deepcopy(SCHEDULE)
    for e in shifted[1:]:e['time_s']+=.5
    waiting=[dict(time_s=0.,mode='stand',heading_deg=0.)]+[dict(e,time_s=e['time_s']+4.) for e in SCHEDULE]
    reversed_turns=copy.deepcopy(SCHEDULE)
    for e in reversed_turns:e['heading_deg']=-e['heading_deg']
    jobs=[('yaw_minus3_confirmation',dict(initial_yaw_deg=-3.),SCHEDULE),
          ('yaw_plus3_confirmation',dict(initial_yaw_deg=3.),SCHEDULE),
          ('xminus3_yawminus1.5',dict(initial_x_offset=-.003,initial_yaw_deg=-1.5),SCHEDULE),
          ('xplus3_yawplus1.5',dict(initial_x_offset=.003,initial_yaw_deg=1.5),SCHEDULE),
          ('friction105',dict(friction_scale=1.05),SCHEDULE),
          ('wait_friction095',dict(friction_scale=.95,seconds=42.),waiting),
          ('reversed_turns',{},reversed_turns),
          ('shifted_yawplus1.5',dict(initial_yaw_deg=1.5),shifted)]
    small_turns=copy.deepcopy(SCHEDULE)
    for e in small_turns:e['heading_deg']*=.75
    jobs += [('new_small_turns_mu098',dict(friction_scale=.98),small_turns),
             ('new_reverse_yawminus1.5',dict(initial_yaw_deg=-1.5),reversed_turns)]
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'plan.json').write_text(json.dumps(jobs,indent=2))
    with ProcessPoolExecutor(2) as pool:out=list(pool.map(trial,jobs))
    (ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')

    if not all(row['result']['passed'] for row in out):raise SystemExit(1)
