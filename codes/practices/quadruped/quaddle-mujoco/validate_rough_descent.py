"""Select on pilot conditions, then test additional resets without changing thresholds."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case
from search_contact_phase import rank
ROOT=Path('outputs/rough_descent_20261110')
def trial(job):
 label,scene,cfg,x,yaw,friction,seconds=job
 r=run_case(scene,'walk',config_override=cfg,initial_x_offset=x,initial_yaw_deg=yaw,friction_scale=friction,seconds=seconds,output=ROOT/'validation'/label)
 print(label,r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
 return dict(label=label,result=r)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=['stairs_down','rough'],required=True);args=p.parse_args()
 rows=json.loads((ROOT/'search.json').read_text())
 if args.scene=='stairs_down':
  rows=[x for x in rows if x['label'].startswith('down_')];assert len(rows)==18
  groups={}
  for x in rows:groups.setdefault(x['label'].rsplit('_',1)[0],[]).append(x)
  def score(pair):return (sum(x['result']['passed'] for x in pair),min(sum(x['result']['checks'].values()) for x in pair),min(x['result']['forward_m'] for x in pair))
  label,pair=max(groups.items(),key=lambda item:score(item[1]));cfg=pair[0]['result']['controller_config']
  selected=dict(label=label,pilot_conditions=pair)
 else:
  rows=[x for x in rows if x['label'].startswith('rough_')];assert len(rows)==12
  best=max(rows,key=lambda x:rank(x['result']));cfg=best['result']['controller_config'];selected=best
 (ROOT/(args.scene+'_selected.json')).write_text(json.dumps(selected,indent=2)+'\n')
 cases=[('xminus',-.005,0.,1.,20),('xplus',.005,0.,1.,20),('yawminus',0.,-3.,1.,20),('yawplus',0.,3.,1.,20),('friction09',0.,0.,.9,20),('friction11',0.,0.,1.1,20),('jointplus',0.,3.,.9,20)]
 if args.scene=='stairs_down':cases.extend([('joint_xminus',-.005,-3.,.9,20),('joint_xplus',.005,-3.,.9,20),('long40',0.,0.,1.,40)])
 with ProcessPoolExecutor(2) as pool:rows=list(pool.map(trial,[(args.scene+'_'+label,args.scene,cfg,x,yaw,friction,seconds) for label,x,yaw,friction,seconds in cases]))
 (ROOT/(args.scene+'_validation.json')).write_text(json.dumps(rows,indent=2)+'\n')
