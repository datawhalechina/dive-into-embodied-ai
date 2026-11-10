"""Measure per-leg support/riser contacts and drive state without changing dynamics."""
import csv,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco
import numpy as np
from scenarios import make_scene,load_scene
from gait_controller import JointController,CONFIG
from run_sim import SIDES


def diagnose(name):
    model,data,rotation=load_scene(make_scene(name));ctrl=JointController('walk',config=dict(CONFIG))
    ids=[model.joint(s+'_thigh').id for s in SIDES];qids=model.jnt_qposadr[ids];vids=model.jnt_dofadr[ids]
    feet=[{i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(s+'_wheel') and '_contact_' in (model.geom(i).name or '')} for s in SIDES]
    support={i for i in range(model.ngeom) if model.geom_bodyid[i]==0 and model.geom_contype[i]}
    rows=[]
    for step in range(10000):
        ctrl(data);mujoco.mj_step(model,data)
        if step%5==0:mujoco.mj_forward(model,data)
        if step%10:continue
        for i,side in enumerate(SIDES):
            cs=[c for c in data.contact if (c.geom1 in feet[i] and c.geom2 in support) or (c.geom2 in feet[i] and c.geom1 in support)]
            phase=((data.time-2)*CONFIG['frequency_hz']+CONFIG['phase_rad'][i]/(2*math.pi))%1
            rows.append(dict(t=float(data.time),side=side,phase=phase,support=int(any(abs(c.frame[2])>.7 for c in cs)),
                riser=int(any(abs(c.frame[2])<.4 for c in cs)),q=float(data.qpos[qids[i]]*CONFIG['actuator_signs'][i]),
                dq=float(data.qvel[vids[i]]*CONFIG['actuator_signs'][i]),target=float(data.ctrl[i]*CONFIG['actuator_signs'][i]),
                torque=float(data.actuator_force[i]),forward=-float(data.qpos[1]),upright=float(data.body('body').xmat[8])))
    out=Path('outputs/contact_phase_20261110/diagnosis')/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'contacts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    bins=[]
    for i in range(10):
        rs=[r for r in rows if r['t']>3.5 and i/10<=r['phase']<(i+1)/10]
        bins.append(dict(phase=i/10,support=np.mean([r['support'] for r in rs]),riser=np.mean([r['riser'] for r in rs]),mean_dq=np.mean([r['dq'] for r in rs])))
    first={s:next((r for r in rows if r['side']==s and r['t']>3.5 and r['riser']),None) for s in SIDES}
    result=dict(scene=name,bins=bins,first_riser=first)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    with ProcessPoolExecutor(2) as p:
        for r in p.map(diagnose,['flat','stairs_up']):print(json.dumps(r),flush=True)
