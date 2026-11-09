"""Measure the untrained periodic controller on each curriculum level."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from curriculum_rl import curriculum_scene
from scenario_matrix import run_case


def job(level):
    return run_case('rough','walk',output=Path('outputs/curriculum/baseline')/f'level{level}',
        scene_definition=curriculum_scene('rough',level))

if __name__=='__main__':
    with ProcessPoolExecutor(2) as pool:results=list(pool.map(job,[1,2,3]))
    Path('outputs/curriculum/baseline/summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for i,r in enumerate(results,1):print(i,r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v])
