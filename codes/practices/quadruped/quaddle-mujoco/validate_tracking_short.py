"""Validate the second nominal passing candidate, independently of distance ranking."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene
ROOT=Path('outputs/foot_trajectory_20261110/short_hold_validation')
def trial(spec):
 label,cfg,scene,definition,x,yaw,friction,seconds=spec
 r=run_case(scene,'walk',config_override=cfg,scene_definition=definition,initial_x_offset=x,initial_yaw_deg=yaw,friction_scale=friction,seconds=seconds,output=ROOT/label)
 print(label,r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,result=r)
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path('outputs/support_coordination_20261110/coupled/selected.json').read_text())['result']['controller_config']
 cfg.update(riser_tracking_limit_rad=.35,tracking_release_s=.04,tracking_scope='all')
 scene=curriculum_scene('stairs_up',2)
 tests=[('offset_minus','stairs_up',scene,-.005,0.,1.,20),('offset_plus','stairs_up',scene,.005,0.,1.,20),('yaw_minus','stairs_up',scene,0.,-3.,1.,20),('yaw_plus','stairs_up',scene,0.,3.,1.,20),('friction09','stairs_up',scene,0.,0.,.9,20),('flat','flat',None,0.,0.,1.,20),('long40','stairs_up',scene,0.,0.,1.,40)]
 with ProcessPoolExecutor(2) as pool:rows=list(pool.map(trial,[(label,cfg,*rest) for label,*rest in tests]))
 (ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

 # Separate boundary diagnostic: original platform ends at 1.8m.
 extended=curriculum_scene('stairs_up',2);extended.boxes[-1]['s1']=4.
 extended.label+=' / extended last platform to 4m'
 r=run_case('stairs_up','walk',seconds=40,config_override=cfg,scene_definition=extended,output=ROOT.parent/'extended_platform40')
 print('extended_platform40',r['passed'],r['forward_m'],flush=True)
