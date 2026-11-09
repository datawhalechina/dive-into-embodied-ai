"""Small reset-position sensitivity check, not multi-seed training robustness."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scenario_matrix import run_case


def trial(job):
    scene,offset=job
    config=json.loads((Path('outputs/terrain_ppo/search')/scene/'selected.json').read_text())
    return run_case(scene,'walk',output=Path('outputs/terrain_ppo/offsets')/str(offset),
        config_override=config,initial_x_offset=offset)

if __name__=='__main__':
    jobs=[(s,x) for s in ['downhill3','stairs_down','uphill5'] for x in [-.005,.005]]
    with ProcessPoolExecutor(2) as pool:
        results=list(pool.map(trial,jobs))
    Path('outputs/terrain_ppo/offsets/summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results: print(r['scene']['name'],r['initial_x_offset_m'],r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v])
