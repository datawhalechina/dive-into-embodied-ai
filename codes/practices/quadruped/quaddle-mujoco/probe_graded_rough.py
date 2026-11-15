"""Evaluate transferred controllers on separate bounded-jump rough terrain."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from scenarios import make_scene
from optimize_rough_descent import contact_config
from graded_rough import graded_rough,geometry_stats
ROOT=Path('outputs/rough_descent_20261110/graded')
def trial(spec):
 label,seed,cfg,policy=spec;scene=graded_rough(seed)
 r=run_case('rough','walk',config_override=cfg,policy_path=policy,scene_definition=scene,output=ROOT/label)
 print(label,r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,geometry=geometry_stats(scene),result=r)
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 (ROOT/'original_geometry.json').write_text(json.dumps(geometry_stats(make_scene('rough')),indent=2)+'\n')
 jobs=[(f'contact_seed{s}',s,contact_config(),None) for s in [7,19,23]]
 jobs.append(('prior_ppo_seed7',7,None,Path('outputs/practice_20261110/rough/prior_policy')))
 with ProcessPoolExecutor(2) as pool:rows=list(pool.map(trial,jobs))
 (ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
