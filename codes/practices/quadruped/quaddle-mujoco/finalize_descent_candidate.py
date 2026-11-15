"""Include both yaw directions at low friction before choosing a final candidate."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from optimize_rough_descent import trial,ROOT
from validate_rough_descent import trial as validate
if __name__=='__main__':
 rows=json.loads((ROOT/'descent_margin.json').read_text());assert len(rows)==18
 groups={}
 for x in rows:groups.setdefault(x['label'].rsplit('_',1)[0],[]).append(x)
 jobs=[(label+'_jointplus','stairs_down',pair[0]['result']['controller_config'],3.,.9,0.,None,False) for label,pair in groups.items()]
 with ProcessPoolExecutor(2) as pool:extra=list(pool.map(trial,jobs))
 (ROOT/'descent_margin_jointplus.json').write_text(json.dumps(extra,indent=2)+'\n')
 for x in extra:groups[x['label'].rsplit('_',1)[0]].append(x)
 def score(pair):return (sum(x['result']['passed'] for x in pair),min(sum(x['result']['checks'].values()) for x in pair),min(x['result']['forward_m'] for x in pair))
 label,pair=max(groups.items(),key=lambda x:score(x[1]));cfg=pair[0]['result']['controller_config']
 (ROOT/'descent_final_selected.json').write_text(json.dumps(dict(label=label,pilot_conditions=pair),indent=2)+'\n')
 cases=[('xminus',-.005,0.,1.,20),('xplus',.005,0.,1.,20),('yawminus',0.,-3.,1.,20),('yawplus',0.,3.,1.,20),('friction09',0.,0.,.9,20),('friction11',0.,0.,1.1,20),('jointplus',0.,3.,.9,20),('joint_xminus',-.005,-3.,.9,20),('joint_xplus',.005,-3.,.9,20),('long40',0.,0.,1.,40)]
 with ProcessPoolExecutor(2) as pool:checks=list(pool.map(validate,[('final_down_'+name,'stairs_down',cfg,x,yaw,friction,seconds) for name,x,yaw,friction,seconds in cases]))
 (ROOT/'descent_final_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
