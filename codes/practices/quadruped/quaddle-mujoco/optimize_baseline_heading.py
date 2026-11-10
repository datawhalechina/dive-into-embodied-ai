"""Reproduce bounded 2mm heading studies; all failures remain in the report.

No candidate from the recorded studies passed both nominal and +3deg starts.
This is an experiment tool, not an automatic baseline promotion mechanism.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene
from command_interaction import CommandController
ROOT=Path(__file__).resolve().parent

def configurations(study):
    base=json.loads((ROOT/'reference/terrain-stairs_up-2mm-previous.json').read_text())
    patches={
      'gains':[dict(heading_kp=kp,lateral_kp=kx) for kp,kx in [(.4,2.),(.4,0.),(.4,1.),(.6,0.),(.6,1.),(.6,2.),(.8,0.),(.8,1.),(.8,2.),(1.,0.),(1.,1.),(.25,0.)]],
      'damping':[dict(heading_kp=kp,heading_kd=kd,lateral_kp=0.) for kp,kd in [(.5,.03),(.55,.03),(.65,.03),(.7,.03),(.6,0.),(.6,.01),(.6,.06),(.6,.1),(.4,.06),(.4,.1)]],
      'limit':[dict(heading_kp=kp,lateral_kp=kx,steering_limit=limit) for kp,kx in [(.4,2.),(.6,0.)] for limit in [.15,.25,.3,.4]],
      'integral':[dict(heading_kp=kp,lateral_kp=kx,heading_ki=ki) for kp,kx,ki in [(.4,2.,.03),(.4,2.,.06),(.4,2.,.12),(.4,0.,.06),(.6,0.,.03),(.6,0.,.06),(.6,0.,.12),(.6,1.,.06)]],
      'frequency':[dict(frequency_hz=f) for f in [1.6,1.65,1.7,1.75,1.85,1.9]],
      'filter':[dict(heading_kp=kp,lateral_kp=kx,heading_filter_s=tau) for kp,kx in [(.4,2.),(.6,0.)] for tau in [.03,.06,.12,.25]],
      'trim':[dict(heading_target_rad=math.radians(deg)) for deg in [-3.,-1.5,1.5,3.]],
    }
    return [dict(base,**patch) for patch in patches[study]]

def trial(job):
    study,i,cfg,yaw,output=job
    r=run_case('stairs_up','walk',config_override=cfg,scene_definition=curriculum_scene('stairs_up',2),initial_yaw_deg=yaw,output=output/study/str(i)/str(yaw))
    print(study,i,yaw,r['passed'],round(r['forward_m'],3),round(r['max_heading_deviation_deg'],2),flush=True)
    return dict(study=study,candidate=i,yaw=yaw,result=r)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['all','gains','damping','limit','integral','frequency','filter','trim'],default='gains')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/heading-study')
    args=p.parse_args()
    studies=['gains','damping','limit','integral','frequency','filter','trim'] if args.study=='all' else [args.study]
    jobs=[(study,i,cfg,yaw,args.output) for study in studies for i,cfg in enumerate(configurations(study)) for yaw in [0.,3.]]
    with ProcessPoolExecutor(2) as pool:rows=list(pool.map(trial,jobs))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
