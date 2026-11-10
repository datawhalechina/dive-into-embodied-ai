"""Limit tracking error on riser contact and briefly after release; unchanged physics."""
import json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene
from search_contact_phase import rank
ROOT=Path('outputs/foot_trajectory_20261110/search')
def trial(spec):
 label,cfg,definition,offset,yaw,friction=spec
 r=run_case('stairs_up','walk',config_override=cfg,scene_definition=definition,initial_x_offset=offset,initial_yaw_deg=yaw,friction_scale=friction,output=ROOT/label)
 print(label,r['passed'],round(r['forward_m'],3),round(r['max_driven_joint_speed_rad_s'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,result=r)
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 base=json.loads(Path('outputs/support_coordination_20261110/coupled/selected.json').read_text())['result']['controller_config']
 candidates=[('baseline',base)]
 for limit in [.15,.25,.35]:
  for hold in [.04,.12]:
   for scope in ['rear','all']:
    candidates.append((f'lag{limit}_hold{hold}_{scope}',dict(base,riser_tracking_limit_rad=limit,tracking_release_s=hold,tracking_scope=scope)))
 jobs=[]
 for level,definition in [('2mm',curriculum_scene('stairs_up',2)),('single4mm',curriculum_scene('stairs_up',3,single_step=True))]:
  jobs.extend((level+'_'+label,cfg,definition,0.,0.,1.) for label,cfg in candidates)
 source=ROOT/'source';source.mkdir(exist_ok=True);hashes={}
 for name in ['contact_phase_controller.py','support_coordinator.py','search_tracking_release.py','diagnose_foot_trajectory.py','scenario_matrix.py','gait_controller.py','gait.json','scenarios.py']:
  b=Path(name).read_bytes();(source/name).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
 (ROOT/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 rows=[]
 with ProcessPoolExecutor(2) as pool:
  for x in pool.map(trial,jobs):
   rows.append(x);(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 selected={}
 for level in ['2mm','single4mm']:
  selected[level]=max([x for x in rows if x['label'].startswith(level+'_') and not x['label'].endswith('baseline')],key=lambda x:rank(x['result']))
 (ROOT/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
 cfg=selected['2mm']['result']['controller_config'];definition=curriculum_scene('stairs_up',2)
 tests=[('offset_minus',cfg,definition,-.005,0.,1.),('offset_plus',cfg,definition,.005,0.,1.),('yaw_minus',cfg,definition,0.,-3.,1.),('yaw_plus',cfg,definition,0.,3.,1.),('friction09',cfg,definition,0.,0.,.9),('original4mm',cfg,None,0.,0.,1.)]
 with ProcessPoolExecutor(2) as pool:checks=list(pool.map(trial,tests))
 (ROOT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
