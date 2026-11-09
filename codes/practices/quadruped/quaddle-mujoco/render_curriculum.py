"""Render the same checkpoint on easy and original terrain, plus single steps."""
from pathlib import Path
from curriculum_rl import curriculum_scene
from scenario_matrix import run_case

if __name__ == '__main__':
    root=Path('outputs/curriculum/videos')
    policy=Path('outputs/curriculum/rough_g999/level1_steps16384')
    cases=[('rough_easy','rough',curriculum_scene('rough',1),policy),
           ('rough_original','rough',curriculum_scene('rough',3),policy),
           ('step_1mm','stairs_up',curriculum_scene('stairs_up',1,single_step=True),None),
           ('step_4mm','stairs_up',curriculum_scene('stairs_up',3,single_step=True),None)]
    for label,name,scene,checkpoint in cases:
        r=run_case(name,'walk',output=root/label,video=True,policy_path=checkpoint,scene_definition=scene)
        print(label,r['passed'],r['forward_m'],flush=True)
