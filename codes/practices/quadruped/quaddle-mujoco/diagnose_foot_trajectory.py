"""Read-only contact/foot trace through the exact matrix integration loop."""
import csv,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mujoco
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene
from run_sim import SIDES
ROOT=Path('outputs/foot_trajectory_20261110')
class FootTrace:
    def __init__(self):self.rows=[];self.peak=None;self.ids=None
    def __call__(self,m,d,c,step):
        if self.ids is None:
            self.ids=[m.jnt_dofadr[m.joint(s+'_thigh').id] for s in SIDES]
            self.qids=[m.jnt_qposadr[m.joint(s+'_thigh').id] for s in SIDES]
            self.feet=[{i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(s+'_wheel') and '_contact_' in (m.geom(i).name or '')} for s in SIDES]
            self.ground={i for i in range(m.ngeom) if m.geom_bodyid[i]==0 and m.geom_contype[i]}
            self.vertices={}
            for ids in self.feet:
                for j in ids:
                    mesh=m.geom_dataid[j];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
                    self.vertices[j]=m.mesh_vert[start:start+count].copy()
        speed=np.abs(d.qvel[self.ids]);new_peak=self.peak is None or speed.max()>self.peak['speed_rad_s']
        if step%5 and not new_peak:return
        rows=[]
        for i,side in enumerate(SIDES):
            support=riser=force=0.
            for j,x in enumerate(d.contact):
                if not ((x.geom1 in self.feet[i] and x.geom2 in self.ground) or (x.geom2 in self.feet[i] and x.geom1 in self.ground)):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f)
                force+=max(0.,float(f[0]));support+=int(abs(x.frame[2])>.7);riser+=int(abs(x.frame[2])<.4)
            points=[]
            for j in self.feet[i]:
                rot=d.geom_xmat[j].reshape(3,3);vertices=self.vertices[j]
                vertex=vertices[np.argmin(vertices@rot[2])]
                points.append(d.geom_xpos[j]+rot@vertex)
            pos=min(points,key=lambda x:x[2])
            phase=float(c.coordinator.phase[i]) if c.coordinator and c.coordinator.phase is not None else None
            r=dict(time_s=float(d.time),leg=side,q=float(d.qpos[self.qids[i]]),dq=float(d.qvel[self.ids[i]]),target=float(d.ctrl[i]),torque_Nm=float(d.actuator_force[i]),normal_force_N=force,support=int(support),riser=int(riser),phase=phase,foot_x_m=float(pos[0]),foot_forward_m=-float(pos[1]),foot_bottom_z_m=float(pos[2]),body_forward_m=-float(d.qpos[1]),upright=float(d.body('body').xmat[8]),yaw_deg=float(np.degrees(np.arctan2(d.body('body').xmat[3],d.body('body').xmat[0]))),pitch_deg=float(np.degrees(np.arctan2(-d.body('body').xmat[7],np.linalg.norm(d.body('body').xmat.reshape(3,3)[:2,1])))),geometry_refreshed=step%5==0)
            rows.append(r)
        if new_peak:
            self.peak=dict(speed_rad_s=float(speed.max()),sample=rows[int(speed.argmax())])
        if step%5==0:self.rows.extend(rows)
    def save(self,path):
        path.mkdir(parents=True,exist_ok=True)
        with (path/'feet.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=self.rows[0]);w.writeheader();w.writerows(self.rows)
        (path/'peak.json').write_text(json.dumps(self.peak,indent=2)+'\n')

def diagnose(level):
    cfg=json.loads(Path('outputs/support_coordination_20261110/coupled/selected.json').read_text())['result']['controller_config']
    trace=FootTrace();out=ROOT/level
    scene=curriculum_scene('stairs_up',2) if level=='stairs_2mm' else curriculum_scene('stairs_up',3,single_step=True)
    r=run_case('stairs_up','walk',config_override=cfg,scene_definition=scene,output=out,observer=trace)
    trace.save(out)
    assert trace.peak['speed_rad_s']==r['max_driven_joint_speed_rad_s']
    print(level,json.dumps(trace.peak),flush=True)
if __name__=='__main__':
    with ProcessPoolExecutor(2) as p:list(p.map(diagnose,['stairs_2mm','single_4mm']))
