"""Locate first physical failure without modifying baseline control or geometry."""
import json, math
from pathlib import Path
import mujoco
import numpy as np
from gait_controller import JointController, CONFIG
from scenarios import make_scene, load_scene
from run_sim import SIDES


def diagnose(name):
    scene=make_scene(name); model,data,rotation=load_scene(scene)
    controller=JointController('walk',config=dict(CONFIG))
    driven=[model.jnt_dofadr[model.joint(s+'_thigh').id] for s in SIDES]
    first={}
    for step in range(22000):
        controller(data);mujoco.mj_step(model,data)
        if step%5: continue
        mujoco.mj_forward(model,data)
        rot=rotation.T@data.body('body').xmat.reshape(3,3)
        yaw=math.degrees(math.atan2(rot[1,0],rot[0,0]))
        names=[(model.geom(c.geom1).name,model.geom(c.geom2).name) for c in data.contact]
        issues={'upright':rot[2,2]<=.8,'heading':abs(yaw)>=25,
            'joint_speed':max(abs(data.qvel[driven]))>10,
            'body_contact':any('body_contact' in pair for pair in names)}
        for key,bad in issues.items():
            if bad and key not in first:
                first[key]={'time_s':float(data.time),'position':data.qpos[:3].tolist(),
                    'yaw_deg':yaw,'upright':float(rot[2,2]),'driven_speed':data.qvel[driven].tolist(),
                    'contacts':names,'ctrl':data.ctrl.tolist()}
    return first

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    names=['stairs_up','stairs_down','rough','downhill3','uphill5']
    with ProcessPoolExecutor(2) as p: results=dict(zip(names,p.map(diagnose,names)))
    output=Path('outputs/terrain_ppo/diagnosis.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
