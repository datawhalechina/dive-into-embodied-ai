"""Hold the existing PPO fixed; evaluate bounded-jump terrain/reset variations."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from graded_rough import graded_rough,geometry_stats
ROOT=Path('outputs/rough_descent_20261110/graded_validation')
POLICY=Path('outputs/practice_20261110/rough/prior_policy')
def trial(spec):
 label,seed,x,yaw,friction=spec;scene=graded_rough(seed)
 r=run_case('rough','walk',policy_path=POLICY,scene_definition=scene,initial_x_offset=x,initial_yaw_deg=yaw,friction_scale=friction,output=ROOT/label)
 print(label,r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,geometry=geometry_stats(scene),result=r)
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 tests=[('seed19',19,0.,0.,1.),('seed23',23,0.,0.,1.),('xminus',7,-.005,0.,1.),('xplus',7,.005,0.,1.),('yawminus',7,0.,-3.,1.),('yawplus',7,0.,3.,1.),('friction09',7,0.,0.,.9)]
 with ProcessPoolExecutor(2) as pool:rows=list(pool.map(trial,tests))
 (ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
