"""Check startup parity, held actions, deterministic replay and bounded target rate."""
import copy
import mujoco
import numpy as np
from contact_phase_controller import ContactPhaseController
from gait_controller import JointController,CONFIG
from scenarios import make_scene,load_scene

cfg=copy.deepcopy(CONFIG);cfg.update(contact_phase_controller=True,contact_adaptation=True)
m1,d1,_=load_scene(make_scene('stairs_up'));m2,d2,_=load_scene(make_scene('stairs_up'))
a=JointController('walk',config=cfg);b=ContactPhaseController(m2,cfg)
for step in range(2000):
    a(d1);b(d2);mujoco.mj_step(m1,d1);mujoco.mj_step(m2,d2)
    if step%5==0:mujoco.mj_forward(m1,d1);mujoco.mj_forward(m2,d2)
np.testing.assert_array_equal(d1.qpos,d2.qpos)
print('original 2s physical startup parity PASS')
# Replay from reset to check controller state and physical determinism.
states=[]
for repeat in range(2):
    m,d,_=load_scene(make_scene('stairs_up'));c=ContactPhaseController(m,cfg)
    old=np.zeros(4)
    for step in range(6500):
        before=d.qpos.copy();vel=d.qvel.copy();c(d)
        np.testing.assert_array_equal(d.qpos,before);np.testing.assert_array_equal(d.qvel,vel)
        assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        if step>=2000:assert np.max(np.abs(d.ctrl-old))<=.080000001
        old=d.ctrl.copy();mujoco.mj_step(m,d)
        if step%5==0:mujoco.mj_forward(m,d)
    assert c.maximum_shift<=.08 and c.maximum_slew<=8.0000001
    assert any(e['event']=='riser' for e in c.events)
    assert any(e['event']=='touchdown' for e in c.events)
    states.append(d.qpos.copy())
np.testing.assert_array_equal(*states)
print('repeatable contact events / bounded target slew and phase / no state editing or external force PASS')
cfg=dict(CONFIG,contact_phase_controller=True,contact_adaptation=False,trajectory_shape='cosine')
m1,d1,_=load_scene(make_scene('stairs_up'));m2,d2,_=load_scene(make_scene('stairs_up'))
a=JointController('walk',config=cfg);b=ContactPhaseController(m2,cfg)
for step in range(6500):
    a(d1);b(d2);mujoco.mj_step(m1,d1);mujoco.mj_step(m2,d2)
    if step%5==0:mujoco.mj_forward(m1,d1);mujoco.mj_forward(m2,d2)
np.testing.assert_array_equal(d1.qpos,d2.qpos)
np.testing.assert_array_equal(d1.qvel,d2.qvel)
print('exact cosine/no-adaptation physical control group parity PASS')
