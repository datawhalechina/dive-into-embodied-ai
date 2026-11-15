"""Observer parity against archived runs and controller write/target-rate invariants."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scenarios import make_scene,load_scene
from contact_phase_controller import ContactPhaseController
root=Path('outputs/foot_trajectory_20261110')
previous=json.loads(Path('outputs/support_coordination_20261110/coupled/validation.json').read_text())
for name in ['stairs_2mm','single_4mm']:
 expected=next(x['result'] for x in previous if x['label']==name)
 observed=json.loads((root/name/'walk/stairs_up/metrics.json').read_text())
 for key in ['forward_m','checks','max_driven_joint_speed_rad_s','max_heading_deviation_deg','body_contact_samples']:
  assert expected[key]==observed[key],(name,key)
print('read-only observer preserves exact acceptance metrics PASS',flush=True)
cfg=json.loads(Path('outputs/support_coordination_20261110/coupled/selected.json').read_text())['result']['controller_config']
cfg.update(riser_tracking_limit_rad=.15,tracking_release_s=.04,tracking_scope='all')
m,d,_=load_scene(make_scene('stairs_up'));c=ContactPhaseController(m,cfg);old=np.zeros(4)
for step in range(8000):
 q,v=d.qpos.copy(),d.qvel.copy();c(d)
 np.testing.assert_array_equal(q,d.qpos);np.testing.assert_array_equal(v,d.qvel)
 assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
 if step>=2000:assert np.max(np.abs(d.ctrl-old))<=.050000001
 old=d.ctrl.copy();mujoco.mj_step(m,d)
 if step%5==0:mujoco.mj_forward(m,d)
assert c.tracking_clamp_ticks>0
print('tracking correction activates / target slew bounded / no state or force injection PASS',flush=True)
