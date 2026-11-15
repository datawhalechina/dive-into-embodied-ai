"""Check actual dynamics and command isolation, including initially stopped input."""
import json
from pathlib import Path
import mujoco
import numpy as np
from command_interaction import CommandController
from scenarios import load_scene,make_scene
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene

m,d,_=load_scene(make_scene('flat'))
schedule=[dict(time_s=0.,mode='stand',heading_deg=0.),dict(time_s=2.,mode='walk',heading_deg=20.),dict(time_s=7.,mode='stand',heading_deg=20.)]
c=CommandController(schedule=schedule)
for step in range(10000):
    q,v=d.qpos.copy(),d.qvel.copy();previous=d.ctrl.copy()
    c(d)
    np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
    assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
    assert abs(c.integral)<=.25
    if c.inner.mode=='stand':assert np.max(np.abs(d.ctrl-previous))<=.002000001
    mujoco.mj_step(m,d)
    if step%5==0:mujoco.mj_forward(m,d)
assert len(c.events)==3
for mode,heading in [('jump',0.),('walk',float('nan')),('walk',90.)]:
    try:c.command(mode,heading,d.time)
    except ValueError:pass
    else:raise AssertionError('Invalid command accepted')
print('Command isolation / support-stop slew / bounded integral / input rejection PASS',flush=True)
# Adding opt-in integral/factory must preserve the old stair candidate.
rows=json.loads(Path('reports/foot_trajectory_2026-11-10.json').read_text())['search']
old=next(x for x in rows if x['label']=='2mm_lag0.35_hold0.04_all')
r=run_case('stairs_up','walk',config_override=old['controller_config'],scene_definition=curriculum_scene('stairs_up',2),output='outputs/baseline_20261110/legacy_parity')
for key in ['forward_m','checks','max_driven_joint_speed_rad_s','max_heading_deviation_deg','body_contact_samples']:
    assert old[key]==r[key],(key,old[key],r[key])
print('Legacy default path and 2mm candidate acceptance parity PASS',flush=True)
