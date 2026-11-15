"""Measure residual saturation and slew clipping without altering dynamics."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scenarios import make_scene,load_scene
from scenario_matrix import profile
from terrain_rl import PolicyController
from curriculum_rl import CurriculumPolicyController

p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--scene',default='rough');a=p.parse_args()
meta=json.loads((a.checkpoint/'metadata.json').read_text())
scene=make_scene(a.scene);model,data,rotation=load_scene(scene)
cls=CurriculumPolicyController if meta.get('environment_version')==2 else PolicyController
controller=cls(model,scene,rotation,profile('walk'),a.checkpoint)
actions=[];clipped=[];later=[]
for step in range(22000):
    controller(data)
    if step>=2000 and step%10==0:
        desired=np.clip(controller.reference+controller.scale*controller.action,-1.2,1.2)
        actions.append(controller.action.copy());clipped.append(np.abs(desired-data.ctrl)>1e-8)
        if data.time>=8:later.append(controller.action.copy())
    mujoco.mj_step(model,data)
    if step%5==0:mujoco.mj_forward(model,data)
actions=np.array(actions);later=np.array(later)
result={'checkpoint':str(a.checkpoint),'max_abs_action_per_joint':np.max(np.abs(actions),axis=0).tolist(),
    'saturated_component_fraction':float(np.mean(np.abs(actions)>=.95)),
    'slew_clipped_component_fraction':float(np.mean(clipped)),
    'late_mean_action':later.mean(axis=0).tolist(),'final_y_m':float(data.qpos[1])}
(a.checkpoint/'action_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
