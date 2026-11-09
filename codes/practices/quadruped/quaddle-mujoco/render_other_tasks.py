"""Render bounded-pilot outcomes, preserving original full-difficulty failures."""
import argparse,json
from pathlib import Path
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene

if __name__=='__main__':
    root=Path('outputs/other_tasks');out=root/'videos'
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['all','stairs','running'],default='all');args=parser.parse_args()
    if args.part in ('all','stairs'):
        progress=json.loads((root/'stairs_curriculum/progress.json').read_text())
        checkpoint=Path(progress['best_checkpoint'])
        meta=json.loads((checkpoint/'metadata.json').read_text())
        for label,scene in [('stairs_course',curriculum_scene('stairs_up',meta['level'])),('stairs_original',None)]:
            r=run_case('stairs_up','walk',output=out/label,video=True,policy_path=checkpoint,scene_definition=scene)
            print(label,r['passed'],r['forward_m'],flush=True)
    if args.part in ('all','running'):
        config=json.loads((root/'run_refinement/selected.json').read_text())
        r=run_case('flat','run_attempt',output=out/'running_selected',video=True,config_override=config)
        print('running_selected',r['passed'],r['forward_m'],flush=True)
