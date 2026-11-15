"""Increase progress margin while jointly retaining nominal/low-friction/yaw checks."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from optimize_rough_descent import trial,descent_config,ROOT
if __name__=='__main__':
 jobs=[]
 for f in [1.04,1.06]:
  for kp in [.25,.3,.35]:
   cfg=descent_config();cfg.update(frequency_hz=cfg['frequency_hz']*f,heading_kp=kp)
   for tag,yaw,friction in [('nominal',0.,1.),('joint',-3.,.9),('friction',0.,.9)]:
    jobs.append((f'down_margin_f{f}_kp{kp}_{tag}','stairs_down',cfg,yaw,friction,0.,None,False))
 rows=[]
 with ProcessPoolExecutor(2) as pool:
  for x in pool.map(trial,jobs):
   rows.append(x);(ROOT/'descent_margin.json').write_text(json.dumps(rows,indent=2)+'\n')
 groups={}
 for x in rows:groups.setdefault(x['label'].rsplit('_',1)[0],[]).append(x)
 score=lambda pair:(sum(x['result']['passed'] for x in pair),min(sum(x['result']['checks'].values()) for x in pair),min(x['result']['forward_m'] for x in pair))
 label,pair=max(groups.items(),key=lambda x:score(x[1]))
 (ROOT/'descent_margin_selected.json').write_text(json.dumps(dict(label=label,pilot_conditions=pair),indent=2)+'\n')
