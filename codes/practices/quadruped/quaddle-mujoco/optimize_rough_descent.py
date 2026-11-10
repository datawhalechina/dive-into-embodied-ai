"""Diagnose rough ground and descending stairs using the unchanged matrix checks."""
import json,hashlib,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case,profile
from diagnose_foot_trajectory import FootTrace
ROOT=Path('outputs/rough_descent_20261110')
def contact_config():
 rows=json.loads(Path('outputs/foot_trajectory_20261110/search/summary.json').read_text())
 return next(x['result']['controller_config'] for x in rows if x['label']=='2mm_lag0.35_hold0.04_all').copy()
def descent_config():return json.loads(Path('reference/terrain-stairs_down.json').read_text())
def trial(job):
 label,scene,cfg,yaw,friction,x,policy,trace_enabled=job
 trace=FootTrace() if trace_enabled else None
 r=run_case(scene,'walk',config_override=cfg,initial_yaw_deg=yaw,friction_scale=friction,initial_x_offset=x,policy_path=policy,output=ROOT/label,observer=trace)
 if trace:trace.save(ROOT/label)
 print(label,r['passed'],round(r['forward_m'],3),round(r['max_driven_joint_speed_rad_s'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,result=r)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['diagnose','search'],default='diagnose');args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 if args.stage=='diagnose':
  down=descent_config();jobs=[('down_nominal','stairs_down',down,0.,1.,0.,None,True),('down_joint','stairs_down',down,-3.,.9,0.,None,True),('rough_prior_ppo','rough',None,0.,1.,0.,Path('outputs/practice_20261110/rough/prior_policy'),True),('rough_transfer','rough',contact_config(),0.,1.,0.,None,True)]
 else:
  jobs=[]
  for f in [.98,1.,1.03]:
   for kp in [.3,.4,.5]:
    cfg=descent_config();cfg.update(frequency_hz=cfg['frequency_hz']*f,heading_kp=kp)
    for tag,yaw,friction in [('nominal',0.,1.),('joint',-3.,.9)]:jobs.append((f'down_f{f}_kp{kp}_{tag}','stairs_down',cfg,yaw,friction,0.,None,False))
  base=contact_config()
  for freq in [1.5,1.8]:
   for kp in [.3,.5,.7]:
    for lag in [.25,.35]:
     cfg=dict(base,frequency_hz=freq,heading_kp=kp,riser_tracking_limit_rad=lag)
     jobs.append((f'rough_f{freq}_kp{kp}_lag{lag}','rough',cfg,0.,1.,0.,None,False))
 source=ROOT/(args.stage+'_source');source.mkdir(exist_ok=True);hashes={}
 for name in ['optimize_rough_descent.py','diagnose_foot_trajectory.py','contact_phase_controller.py','support_coordinator.py','scenario_matrix.py','gait_controller.py','scenarios.py','terrain_rl.py','curriculum_rl.py']:
  b=Path(name).read_bytes();(source/name).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
 (source/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 rows=[]
 with ProcessPoolExecutor(2) as pool:
  for row in pool.map(trial,jobs):
   rows.append(row);(ROOT/(args.stage+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
