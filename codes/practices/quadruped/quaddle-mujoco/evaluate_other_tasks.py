"""Repeat accepted walking scenes with both lateral offsets; preserve failures."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case


def trial(job):
    name,x=job
    config_path=Path('reference')/f'terrain-{name}.json'
    config=json.loads(config_path.read_text()) if name in ('uphill5','downhill3','stairs_down') else None
    return run_case(name,'walk',output=Path('outputs/other_tasks/walk_recheck')/f'x{x:+.3f}',
        initial_x_offset=x,config_override=config)

if __name__=='__main__':
    out=Path('outputs/other_tasks/walk_recheck');out.mkdir(parents=True,exist_ok=True)
    names=['flat','uphill3','uphill5','downhill3','stairs_down','avoid','water']
    results=[]
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(trial,[(n,x) for n in names for x in [-.005,.005]]):
            results.append(r);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            print(r['scene']['name'],r['initial_x_offset_m'],r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v],flush=True)
