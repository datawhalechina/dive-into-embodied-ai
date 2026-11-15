"""Reset-only finer terrain sampling with explicit first-violation reward costs."""
from collections import OrderedDict
import math
import gymnasium as gym
import numpy as np
from curriculum_rl import CurriculumEnv,curriculum_scene
from scenarios import load_scene


def practice_scene(name, scale=1., seed=7):
    if not .2 <= scale <= 1.:raise ValueError('terrain scale must be in [.2,1]')
    scene=curriculum_scene(name,3,seed)
    for box in scene.boxes:box['top']*=scale
    scene.plane_z*=scale
    scene.label=f'{name} practice / scale {scale:g} / seed {seed}'
    return scene


class PracticeEnv(CurriculumEnv):
    def __init__(self,scene='rough',seconds=20.,feedback_gain=.02):
        super().__init__(scene,seconds=seconds)
        self.task=scene
        self.config=dict(self.config,speed_feedback_gain=feedback_gain,speed_feedback_threshold=8.)
        self.cache=OrderedDict()

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        scale=float(self.np_random.choice([.25,.375,.5,1.],p=[.3,.3,.3,.1]))
        terrain_seed=int(self.np_random.choice([7,11,13,17])) if self.task=='rough' else 7
        key=(scale,terrain_seed)
        if key not in self.cache:
            scene=practice_scene(self.task,scale,terrain_seed)
            model,data,rotation=load_scene(scene)
            self.cache[key]=(scene,model,data,rotation,data.qpos.copy(),model.geom_friction.copy())
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        self.scene,self.model,self.data,self.rotation,initial,friction=self.cache[key]
        self.initial=initial.copy()
        angle=float(self.np_random.uniform(-math.radians(3),math.radians(3)))
        self.initial[3:7]=[math.cos(angle/2),0.,0.,math.sin(angle/2)]
        friction_scale=float(self.np_random.uniform(.9,1.1))
        self.model.geom_friction[:]=friction
        self.model.geom_friction[:,0]*=friction_scale
        self.body_geom=self.model.geom('body_contact').id
        self.axis=self.rotation@np.array([0.,-1.,0.])
        self.violations=set()
        self.episode_parameters=dict(terrain_scale=scale,terrain_seed=terrain_seed,yaw_rad=angle,friction_scale=friction_scale)
        obs,info=super().reset(seed=None,options=options)
        info.update(self.episode_parameters)
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        for name,bad in [('joint_speed',info['peak_joint_speed']>10.),('heading',abs(info['yaw_deg'])>=25.)]:
            if bad and name not in self.violations:
                reward-=2.;self.violations.add(name)
        reward-=.01*2.*max(0.,abs(info['yaw_deg'])-15.)/10.
        info.update(self.episode_parameters,violations=sorted(self.violations))
        return obs,float(reward),terminated,truncated,info
