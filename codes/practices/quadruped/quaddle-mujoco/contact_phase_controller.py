"""Experimental joint trajectory with bounded contact-triggered phase adjustments.

Uses simulator contact normals, not a hardware contact estimator. Only writes ctrl.
"""
import math
import numpy as np
from gait_controller import JointController
from run_sim import SIDES


class ContactPhaseController:
    def __init__(self,model,config):
        self.model,self.config=model,config
        self.startup=JointController('walk',config=config)
        self.signs=np.array(config['actuator_signs']);self.phase=np.array(config['phase_rad'])/(2*math.pi)
        self.feet=[{i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(s+'_wheel') and '_contact_' in (model.geom(i).name or '')} for s in SIDES]
        self.support={i for i in range(model.ngeom) if model.geom_bodyid[i]==0 and model.geom_contype[i]}
        self.target=np.zeros(4);self.tick=None;self.shift=np.zeros(4);self.boost=np.zeros(4)
        self.previous_contact=np.zeros(4,dtype=bool);self.contact_count=np.zeros(4,dtype=int)
        self.landing_until=np.zeros(4);self.riser_until=np.zeros(4);self.cooldown=np.zeros(4)
        self.events=[];self.maximum_slew=0.;self.maximum_shift=0.;self.start_x=None
        self.tracking_until=np.zeros(4);self.tracking_clamp_ticks=0
        self.qids=[model.jnt_qposadr[model.joint(s+'_thigh').id] for s in SIDES]
        self.vids=[model.jnt_dofadr[model.joint(s+'_thigh').id] for s in SIDES]
        self.coordinator=None
        if config.get('support_coordination'):
            from support_coordinator import SupportCoordinator,CoupledSupportCoordinator
            cls=CoupledSupportCoordinator if config.get('coupled_support_clock') else SupportCoordinator
            self.coordinator=cls(config)

    def __call__(self,data):
        c=self.config
        if self.start_x is None:self.start_x=float(data.qpos[0])
        # Exact legacy path for the cosine/no-feedback control group.
        if self.coordinator is None and not c.get('contact_adaptation',False) and c.get('trajectory_shape','quintic')=='cosine':
            previous=self.target.copy();self.startup(data);self.target=data.ctrl.copy()
            self.maximum_slew=max(self.maximum_slew,float(np.max(np.abs(self.target-previous)))*c['control_hz'])
            return
        if data.time < c['settle_seconds']-1e-8:
            self.startup(data);self.target=data.ctrl.copy();return
        tick=round((data.time-c['settle_seconds'])*c['control_hz'])
        if abs(data.time-c['settle_seconds']-tick/c['control_hz'])>1e-7 or tick==self.tick:
            data.ctrl[:]=self.target;return
        self.tick=tick;dt=1/c['control_hz'];t=tick*dt
        contact=np.zeros(4,dtype=bool);riser=np.zeros(4,dtype=bool)
        for i,feet in enumerate(self.feet):
            cs=[x for x in data.contact if (x.geom1 in feet and x.geom2 in self.support) or (x.geom2 in feet and x.geom1 in self.support)]
            contact[i]=any(abs(x.frame[2])>.7 for x in cs)
            riser[i]=any(abs(x.frame[2])<.4 for x in cs)
        enabled=c.get('contact_adaptation',False) and t>=c['ramp_seconds']
        self.contact_count=np.where(contact,self.contact_count+1,0)
        stable=self.contact_count>=2
        touchdown=stable & ~self.previous_contact
        if enabled:
            for i in range(4):
                if touchdown[i]:
                    self.landing_until[i]=t+c.get('landing_hold_s',.04)
                    self.events.append(dict(time_s=float(data.time),leg=SIDES[i],event='touchdown'))
                if riser[i] and t>=self.cooldown[i]:
                    self.riser_until[i]=t+c.get('riser_hold_s',.08)
                    self.cooldown[i]=t+.25
                    self.events.append(dict(time_s=float(data.time),leg=SIDES[i],event='riser'))
        self.previous_contact=stable
        landing=enabled & (t<self.landing_until);blocked=enabled & (t<self.riser_until)
        rate=np.where(landing,.5,1.)
        rate=np.where(blocked,c.get('riser_phase_rate',.4),rate)
        recovery=np.clip(-self.shift,-.003,.003)
        self.shift=np.clip(self.shift+np.where(landing|blocked,c['frequency_hz']*dt*(rate-1),recovery),-.08,.08)
        self.maximum_shift=max(self.maximum_shift,float(np.max(np.abs(self.shift))))
        self.boost+=.25*(np.where(blocked,c.get('riser_angle_rad',.12),0.)-self.boost)
        cycle=(t*c['frequency_hz']+self.phase+self.shift)%1
        if self.coordinator is not None and t>=c['ramp_seconds']:
            cycle=self.coordinator.advance(cycle,stable,dt,float(data.time))
            if c.get('support_recovery') or c.get('coupled_support_clock'):
                landing=np.array([s=='landing' for s in self.coordinator.state])
        duty=c['sweep_fraction'];u=np.where(cycle<duty,cycle/duty,(cycle-duty)/(1-duty))
        if c.get('trajectory_shape','quintic')=='quintic':
            smooth=u**3*(10-15*u+6*u*u)
            wave=np.where(cycle<duty,1-2*smooth,-1+2*smooth)
        else:wave=np.where(cycle<duty,np.cos(np.pi*u),-np.cos(np.pi*u))
        rot=data.body('body').xmat.reshape(3,3);yaw=math.atan2(rot[1,0],rot[0,0])
        delta_yaw=yaw-c.get('heading_target_rad',0.)
        error=math.atan2(math.sin(delta_yaw),math.cos(delta_yaw))+c['lateral_kp']*(data.qpos[0]-self.start_x)
        steer=np.clip(c['heading_kp']*error+c['heading_kd']*data.qvel[5],-c['steering_limit'],c['steering_limit'])
        angle=np.array(c['bias_rad'])+np.array(c['amplitude_rad'])*(1+steer*self.signs)*wave+self.boost
        ramp=min(1.,t/c['ramp_seconds']);startup=np.array(c['startup_bias_rad'])
        desired=np.clip(self.signs*((1-ramp)*startup+ramp*angle),-1.2,1.2)
        if c.get('riser_tracking_limit_rad',0)>0 and t>=c['ramp_seconds']:
            hit=riser.copy()
            if c.get('tracking_scope','all')=='rear':hit[:2]=False
            self.tracking_until=np.where(hit,t+c.get('tracking_release_s',.08),self.tracking_until)
            active=t<self.tracking_until
            q=data.qpos[self.qids];limit=c['riser_tracking_limit_rad']
            bounded=np.clip(desired,q-limit,q+limit)
            self.tracking_clamp_ticks+=int(np.any(active & (np.abs(bounded-desired)>1e-12)))
            desired=np.where(active,bounded,desired)
        slew=np.where(landing,c.get('landing_slew_rad_s',3.),c['max_target_speed_rad_s'])
        delta=np.clip(desired-self.target,-slew*dt,slew*dt)
        self.maximum_slew=max(self.maximum_slew,float(np.max(np.abs(delta)))/dt)
        self.target+=delta;data.ctrl[:]=self.target

    def diagnostics(self):
        return dict(controller='contact_phase_v1',events=self.events,max_target_slew_rad_s=self.maximum_slew,
            max_phase_shift_cycles=self.maximum_shift,contact_estimation='simulator contact normals',
            support_coordination=self.coordinator.diagnostics() if self.coordinator is not None else None,
            tracking_clamp_ticks=self.tracking_clamp_ticks)
