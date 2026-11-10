"""Check support gating and physical invariants of the experimental scheduler."""
import numpy as np
import mujoco
from support_coordinator import SupportCoordinator
from contact_phase_controller import ContactPhaseController
from gait_controller import CONFIG
from scenarios import load_scene,make_scene

cfg=dict(CONFIG,support_coordination=True,contact_adaptation=False,
         trajectory_shape='quintic',minimum_support_feet=2,maximum_swing_feet=1)
s=SupportCoordinator(cfg)
s.advance(np.array([.69,.4,.3,.2]),np.zeros(4,dtype=bool),.01,0.)
for tick in range(100):s.advance(np.zeros(4),np.zeros(4,dtype=bool),.01,tick*.01)
assert not any(e['event']=='swing_granted' for e in s.events)
assert all(p<.7 for p in s.phase)
# Rear support plus one other support permits a front swing; no overlap admitted.
for tick in range(100):
    s.advance(np.zeros(4),np.ones(4,dtype=bool),.01,1+tick*.01)
    assert sum(x in ('swing','landing') for x in s.state)<=1
assert any(e['event']=='swing_granted' for e in s.events)
assert any(e['event']=='support_confirmed' for e in s.events)
print('no unsupported swing grants / one swing permission / landing confirmation PASS',flush=True)
# Waiting without support retreats instead of granting an unsupported swing.
r=SupportCoordinator(dict(cfg,support_recovery=True))
r.advance(np.array([.69,.4,.3,.2]),np.zeros(4,dtype=bool),.01,0.)
for tick in range(150):r.advance(np.zeros(4),np.zeros(4,dtype=bool),.01,tick*.01)
assert any(e['event']=='support_recovery' for e in r.events)
assert any(e['event']=='recovery_finished' for e in r.events)
assert not any(e['event']=='swing_granted' for e in r.events)
# A contact flicker must not count as four consecutive contact ticks.
r.state=['landing','stance','stance','stance'];r.timer[:]=0;r.contact_seconds[:]=0
before=len(r.events)
for tick in range(6):
    contact=np.array([tick%2==0,False,False,False])
    r.advance(np.zeros(4),contact,.01,2+tick*.01)
assert not any(e['event']=='support_confirmed' for e in r.events[before:])
print('unsupported-wait recovery / continuous contact confirmation PASS',flush=True)
cfg['support_recovery']=True
ends=[]
for repeat in range(2):
    model,data,_=load_scene(make_scene('stairs_up'));c=ContactPhaseController(model,cfg)
    old=np.zeros(4)
    for step in range(7000):
        q,v=data.qpos.copy(),data.qvel.copy();c(data)
        np.testing.assert_array_equal(q,data.qpos);np.testing.assert_array_equal(v,data.qvel)
        assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
        if step>=2000:assert np.max(np.abs(data.ctrl-old))<=cfg['max_target_speed_rad_s']*.01+1e-9
        old=data.ctrl.copy();mujoco.mj_step(model,data)
        if step%5==0:mujoco.mj_forward(model,data)
    assert c.coordinator.events
    ends.append(data.qpos.copy())
np.testing.assert_array_equal(*ends)
print('deterministic physics / bounded target slew / no external forces or state editing PASS',flush=True)
from support_coordinator import CoupledSupportCoordinator
c=CoupledSupportCoordinator(dict(cfg,coupled_support_clock=True))
initial=np.array([.66,.16,.91,.41]);contact=np.zeros(4,dtype=bool)
c.advance(initial,contact,.01,0.)
for tick in range(300):
    phase=c.advance(initial,contact,.01,(tick+1)*.01)
    difference=(phase-phase[0]-(initial-initial[0])+.5)%1-.5
    assert np.max(np.abs(difference))<1e-12
assert c.clock>0 and c.slow_seconds>0
assert any(e['event']=='support_timeout' for e in c.events)
print('coupled relative phases preserved / bounded support wait makes progress PASS',flush=True)
