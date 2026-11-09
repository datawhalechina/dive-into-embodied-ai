"""Measure the untrained periodic controller on each curriculum level."""
from concurrent.futures import ProcessPoolExecutor
import argparse,json
from pathlib import Path
from curriculum_rl import curriculum_scene
from scenario_matrix import run_case


def job(spec):
    scene,level,output=spec
    return run_case(scene,'walk',output=output/f'level{level}',scene_definition=curriculum_scene(scene,level))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['rough','stairs_up'],default='rough')
    p.add_argument('--output',type=Path,default=Path('outputs/curriculum/baseline'));args=p.parse_args()
    with ProcessPoolExecutor(2) as pool:
        results=list(pool.map(job,[(args.scene,level,args.output) for level in [1,2,3]]))
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for i,r in enumerate(results,1):print(i,r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v])
