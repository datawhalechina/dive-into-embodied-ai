"""Separate yaw and friction effects for failed joint-perturbation cases."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from evaluate_operating_range import trial

if __name__=='__main__':
    root=Path('outputs/practice_20261110/operating_range')
    records=json.loads((root/'summary.json').read_text())
    previous=json.loads((root/'isolated.json').read_text()) if (root/'isolated.json').exists() else []
    completed={(x['result']['scene']['name'],x['label']) for x in previous}
    jobs=[]
    for item in records:
        r=item['result']
        if not r['passed'] and item['label']!='long40s':
            name=r['scene']['name'];yaw=r['initial_yaw_deg'];friction=r['friction_scale']
            jobs.extend([(name,f'isolate_yaw{yaw}',yaw,1.,20.),(name,f'isolate_friction{friction}',0.,friction,20.)])
    jobs=[j for j in jobs if (j[0],j[1]) not in completed]
    with ProcessPoolExecutor(2) as pool:results=previous+list(pool.map(trial,jobs))
    (root/'isolated.json').write_text(json.dumps(results,indent=2)+'\n')
    for item in results:
        r=item['result'];print(item['label'],r['scene']['name'],r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v],flush=True)
