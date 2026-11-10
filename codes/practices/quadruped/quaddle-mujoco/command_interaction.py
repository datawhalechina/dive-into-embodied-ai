"""Joint-only stop/walk/heading interaction, with deterministic command replay.

Keyboard: W walk, S stop, A/D heading +/-20 degrees, Q quit.
Heading commands are walking arcs, not an in-place turn or a yaw teleport.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from queue import SimpleQueue
import time
import numpy as np
import mujoco
from gait_controller import JointController
from scenario_matrix import run_case
from scenarios import load_scene, make_scene

SCHEDULE=[dict(time_s=0.,mode='walk',heading_deg=0.),
          dict(time_s=8.,mode='stand',heading_deg=0.),
          dict(time_s=12.,mode='walk',heading_deg=0.),
          dict(time_s=19.,mode='walk',heading_deg=20.),
          dict(time_s=27.,mode='walk',heading_deg=-20.),
          dict(time_s=35.,mode='stand',heading_deg=-20.)]
CONFIG_PATH=Path(__file__).parent/'reference/interaction.json'

def yaw_deg(data):
    r=data.body('body').xmat.reshape(3,3)
    return math.degrees(math.atan2(r[1,0],r[0,0]))

def angle_error(a,b):
    return (a-b+180.)%360.-180.

class CommandController:
    def __init__(self,config=None,schedule=None):
        self.config=copy.deepcopy(config or json.loads(CONFIG_PATH.read_text()))
        self.inner=JointController('stand',config=self.config)
        self.schedule=copy.deepcopy(schedule or [])
        self.next_event=0
        self.heading_deg=0.
        self.reference_heading_deg=0.
        self.events=[]
        self.integral=0.
        self.last_time=None
    def command(self,mode,heading_deg,now):
        if mode not in ('walk','stand') or not math.isfinite(heading_deg):
            raise ValueError('Expected walk/stand and a finite heading')
        if abs(heading_deg)>20.:
            raise ValueError('Baseline heading commands are limited to +/-20 degrees')
        self.heading_deg=heading_deg
        self.inner.mode=mode
        self.config['heading_target_rad']=math.radians(self.heading_deg)
        self.events.append(dict(time_s=float(now),mode=mode,heading_deg=self.heading_deg))
    def __call__(self,data):
        while self.next_event<len(self.schedule) and data.time+1e-9>=self.schedule[self.next_event]['time_s']:
            e=self.schedule[self.next_event]
            self.command(e['mode'],e['heading_deg'],data.time)
            self.next_event+=1
        dt=0. if self.last_time is None else max(0.,float(data.time)-self.last_time)
        self.last_time=float(data.time)
        rate=self.config.get('heading_rate_deg_s',0.)
        if rate>0.:
            limit=rate*dt
            self.reference_heading_deg+=float(np.clip(self.heading_deg-self.reference_heading_deg,-limit,limit))
        else:
            self.reference_heading_deg=self.heading_deg
        if self.inner.mode=='walk':
            error=math.radians(angle_error(yaw_deg(data),self.reference_heading_deg))
            self.integral=float(np.clip(self.integral+self.config.get('heading_ki',0.)*error*dt,-.25,.25))
            self.config['heading_target_rad']=math.radians(self.reference_heading_deg)-self.integral/self.config['heading_kp']
        if self.inner.mode == 'stand':
            # Keep the same support pose used by safe startup. A zero-angle
            # stop followed by rapid support-pose recovery tipped the body.
            target=np.array(self.config['actuator_signs'])*np.array(self.config['startup_bias_rad'])
            previous=data.ctrl.copy()
            self.inner(data)
            data.ctrl[:]=previous+np.clip(target-previous,-2.*.001,2.*.001)
        else:
            self.inner(data)
    def status(self,data):
        return f'{self.inner.mode} target {self.heading_deg:+.0f}deg / yaw {yaw_deg(data):+.1f}deg'

class Trace:
    def __init__(self):self.rows=[]
    def __call__(self,m,d,c,step):
        if step%5==0:
            self.rows.append(dict(t=float(d.time),xy=d.qpos[:2].tolist(),yaw=yaw_deg(d),mode=c.inner.mode,target=c.heading_deg))

def replay(output,video=False,config=None,schedule=None,seconds=38.,**perturbation):
    schedule=copy.deepcopy(SCHEDULE if schedule is None else schedule)
    if not schedule or schedule[0]['time_s']!=0.:
        raise ValueError('Replay schedule must start at time 0')
    times=[e['time_s'] for e in schedule]+[seconds+2.]
    if not all(math.isfinite(t) for t in times) or any(b-a<2. for a,b in zip(times,times[1:])):
        raise ValueError('Replay metric windows require each command to last at least 2 seconds')
    trace=Trace()
    cfg=copy.deepcopy(config or json.loads(CONFIG_PATH.read_text()))
    raw=run_case('flat','walk',seconds=seconds,output=output,video=video,
                 config_override=cfg,observer=trace,
                 controller_factory=lambda m,c:CommandController(c,schedule),**perturbation)
    # Static straight-line goals are inapplicable to intentional stops and turns.
    physical={k:v for k,v in raw['checks'].items() if k not in ('heading','forward_speed','task_goal')}
    segments=[]
    for i,e in enumerate(schedule):
        end=schedule[i+1]['time_s'] if i+1<len(schedule) else seconds+cfg['settle_seconds']
        rows=[r for r in trace.rows if e['time_s']<=r['t']<end]
        tail=[r for r in rows if r['t']>=end-1.]
        settled=[r for r in rows if r['t']>=e['time_s']+1.]
        if e['mode']=='stand':
            drift=max(float(np.linalg.norm(np.array(r['xy'])-settled[0]['xy'])) for r in settled)
            result=dict(stop_drift_after_1s_m=drift,passed=drift<=.02)
        else:
            delta=np.array(rows[-1]['xy'])-rows[0]['xy']
            heading=math.radians(e['heading_deg'])
            progress=float(delta@np.array([math.sin(heading),-math.cos(heading)]))
            error=max(abs(angle_error(r['yaw'],e['heading_deg'])) for r in tail)
            result=dict(progress_along_command_m=progress,final_1s_max_heading_error_deg=error,passed=progress>.10 and error<10.)
        segments.append(dict(command=e,end_s=end,**result))
    report=dict(passed=all(physical.values()) and all(s['passed'] for s in segments),
                physical_checks=physical,segments=segments,config=cfg,perturbation=perturbation,
                joint_peak_rad_s=raw['max_driven_joint_speed_rad_s'],
                note='Intentional turns replace static heading and straight-line distance checks. Stop drift measured after 1s, <=2cm; walk segment progress >10cm and final 1s heading error <10deg. Fixed model; no forces/base-state edits.')
    folder=Path(output)/'walk/flat'
    (folder/'commands.json').write_text(json.dumps(schedule,indent=2)+'\n')
    (folder/'command_trace.json').write_text(json.dumps(trace.rows)+'\n')
    (folder/'interaction_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def interactive():
    import mujoco.viewer
    m,d,_=load_scene(make_scene('flat'))
    c=CommandController(); events=SimpleQueue(); step=0
    from run_sim import camera_and_options
    print(__doc__,flush=True)
    with mujoco.viewer.launch_passive(m,d,key_callback=events.put,show_left_ui=False,show_right_ui=False) as viewer:
        cam,opt=camera_and_options(d)
        viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=cam.distance,cam.azimuth,cam.elevation
        viewer.opt.geomgroup[:]=opt.geomgroup
        while viewer.is_running():
            start=time.monotonic()
            while not events.empty():
                key=events.get()
                if key==ord('Q'):
                    viewer.close()
                    break
                if key in (ord('W'),ord('S'),ord('A'),ord('D')):
                    mode='stand' if key==ord('S') else 'walk'
                    heading=float(np.clip(c.heading_deg+(20 if key==ord('A') else -20 if key==ord('D') else 0),-20.,20.))
                    c.command(mode,heading,d.time)
                    print(c.status(d),flush=True)
            if not viewer.is_running():break
            for _ in range(16):
                c(d);mujoco.mj_step(m,d)
                if step%5==0:mujoco.mj_forward(m,d)
                step+=1
            viewer.cam.lookat[:]=d.body('body').xpos
            viewer.sync(state_only=True)
            time.sleep(max(0.,.016-(time.monotonic()-start)))
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/interaction_live_commands.json').write_text(json.dumps(c.events,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--commands',type=Path,help='Replay a JSON command schedule (time_s, mode, heading_deg)')
    p.add_argument('--seconds',type=float,default=38.,help='Commanded duration, plus 2 seconds of startup')
    p.add_argument('--output',type=Path,default=Path('outputs/baseline_20261110/interaction'))
    args=p.parse_args()
    if args.interactive:interactive()
    else:
        schedule=json.loads(args.commands.read_text()) if args.commands else None
        result=replay(args.output,args.video,schedule=schedule,seconds=args.seconds)
        print(json.dumps(result,indent=2))
        if not result['passed']:raise SystemExit(1)
