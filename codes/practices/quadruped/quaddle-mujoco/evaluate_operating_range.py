"""Stress existing walking configs; keep nominal acceptance thresholds unchanged."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case


def trial(spec):
    scene,label,yaw,friction,seconds=spec
    path=Path('reference')/f'terrain-{scene}.json'
    config=json.loads(path.read_text()) if scene in ('uphill5','downhill3','stairs_down') else None
    result=run_case(scene,'walk',output=Path('outputs/practice_20261110/operating_range')/label,
        config_override=config,initial_yaw_deg=yaw,friction_scale=friction,seconds=seconds)
    return dict(label=label,result=result)

if __name__=='__main__':
    root=Path('outputs/practice_20261110/operating_range');root.mkdir(parents=True,exist_ok=True)
    names=['flat','uphill3','uphill5','downhill3','stairs_down','avoid','water']
    jobs=[(n,f'yaw{yaw:+}_friction{friction}',yaw,friction,20.) for n in names for yaw,friction in [(-3.,.9),(3.,1.1)]]
    jobs += [(n,'long40s',0.,1.,40.) for n in ['flat','uphill5','stairs_down']]
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for x in pool.map(trial,jobs):
            results.append(x);(root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            r=x['result'];print(r['scene']['name'],x['label'],r['passed'],round(r['forward_m'],3),[k for k,v in r['checks'].items() if not v],flush=True)
