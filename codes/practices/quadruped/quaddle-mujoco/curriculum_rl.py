"""Versioned terrain curriculum; original full-difficulty acceptance is unchanged."""
import math
import json
from pathlib import Path
import numpy as np
from gymnasium import spaces
from scenarios import make_scene
from terrain_rl import TerrainEnv, ResidualController


def curriculum_scene(name, level=3, terrain_seed=7, single_step=False):
    if name not in ('rough', 'stairs_up') or level not in (1, 2, 3):
        raise ValueError('rough/stairs_up with level 1..3 required')
    scene = make_scene(name)
    if name == 'rough':
        rng = np.random.default_rng(terrain_seed)
        scene.seed = terrain_seed
        for box in scene.boxes:
            if box['name'].startswith('tile_'):
                box['top'] = float(rng.uniform(-.003, .003))*(level/3)
        scene.plane_z *= level/3
        scene.label = f'Rough curriculum / +/-{level} mm / seed {terrain_seed}'
    else:
        height = {1:.001, 2:.002, 3:.004}[level]
        for i, box in enumerate(scene.boxes):
            box['top'] = height*(i+1)
        if single_step:
            scene.boxes = [scene.boxes[0]]
            scene.boxes[0]['s1'] = 1.8
            scene.goal_m = .48
        scene.label = f'Stairs curriculum / {height*1000:g} mm / {len(scene.boxes)} steps'
    return scene


def probe_points(data):
    rotation = data.body('body').xmat.reshape(3,3)
    yaw = math.atan2(rotation[1,0],rotation[0,0])
    c,s = math.cos(yaw),math.sin(yaw)
    xy = data.qpos[:2]
    return np.array([xy+np.array([c*x+s*f,s*x-c*f])
        for f in [-.04,0.,.04,.08,.12] for x in [-.055,0.,.055]])


class GridController(ResidualController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.history = np.zeros(8)
        self.obs_tick = None
        self.cached_obs = None
        self.maximum_forward = 0.
        self.origin_y = None
        self.foot_ids = [{i for i in range(self.model.ngeom) if
            (self.model.geom(i).name or '').startswith(side+'_wheel') and '_contact_' in (self.model.geom(i).name or '')}
            for side in ['left_front','right_front','right_back','left_back']]
        self.support_ids = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==0 and self.model.geom_contype[i]}

    def observation(self,data):
        tick=round(data.time*100)
        if tick==self.obs_tick:
            return self.cached_obs.copy()
        if self.origin_y is None:
            self.origin_y=float(data.qpos[1])
        forward=self.origin_y-float(data.qpos[1])
        self.maximum_forward=max(self.maximum_forward,forward)
        base=super().observation(data)[:31]
        heights=[(self.scene.surface_height(x,y)-data.qpos[2])/.05 for x,y in probe_points(data)]
        pairs=[{c.geom1,c.geom2} for c in data.contact if c.geom1 in self.support_ids or c.geom2 in self.support_ids]
        contacts=[float(any(ids & pair for pair in pairs)) for ids in self.foot_ids]
        result=np.concatenate((base,heights,contacts,self.history,
            [(self.scene.goal_m-forward),self.maximum_forward-forward])).astype(np.float32)
        self.history=np.concatenate((data.qpos[self.qids],data.qvel[self.vids]/10.))
        self.obs_tick=tick;self.cached_obs=result
        return result.copy()


class CurriculumEnv(TerrainEnv):
    controller_class=GridController
    def __init__(self, scene='rough', level=1, terrain_seed=7, seed_noise=True, seconds=20.):
        super().__init__(scene,seconds,seed_noise,
            scene_definition=curriculum_scene(scene,level,terrain_seed))
        self.observation_space=spaces.Box(-np.inf,np.inf,(60,),np.float32)

    def reset(self,**kwargs):
        self.milestone=0
        self.goal_awarded=False
        return super().reset(**kwargs)

    def step(self,action):
        previous=self.controller.action.copy()
        before_y=float(self.data.qpos[1])
        obs,_,terminated,truncated,info=super().step(action)
        # Previous max progress is encoded in the observation. Milestones pay once.
        forward=info['forward_m']
        milestones=max(0,int(min(forward,self.scene.goal_m)/.12))
        bonus=.3*max(0,milestones-self.milestone)
        self.milestone=max(self.milestone,milestones)
        goal=forward>=self.scene.goal_m
        if goal and not self.goal_awarded:
            bonus+=2.;self.goal_awarded=True
        yaw=math.radians(info['yaw_deg'])
        # This 10 ms average speed matches displacement, rather than foot oscillation.
        speed=(before_y-float(self.data.qpos[1]))/.01
        reward=.01*(2*np.clip(speed/.075,-2,2)-.15
            -.8*(yaw/.436)**2-.2*(self.data.qpos[0]/.1)**2
            -2.*max(0.,info['peak_joint_speed']-9.)**2
            -.05*np.mean(np.asarray(action)**2)-.1*np.mean((np.asarray(action)-previous)**2))
        reward+=bonus-(4. if terminated else 0.)
        info.update(milestone=self.milestone,goal_reached=goal)
        return obs,float(reward),terminated,truncated,info


class CurriculumPolicyController(GridController):
    def __init__(self,model,scene,rotation,config,directory):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        directory=Path(directory)
        meta=json.loads((directory/'metadata.json').read_text())
        config={**meta.get('controller_config', {}), **config}
        super().__init__(model,scene,rotation,config)
        if meta['environment_version']!=2 or meta['observation_dim']!=60:
            raise ValueError('Incompatible policy observation version')
        self.policy=PPO.load(directory/'policy.zip',device='cpu')
        with np.load(directory/'obs_stats.npz') as stats:
            self.mean,self.var=stats['mean'].copy(),stats['var'].copy()
        self.policy_tick=None

    def __call__(self,data):
        tick=round(data.time*100)
        if data.time>=2.-1e-8 and abs(data.time-tick*.01)<1e-7 and tick!=self.policy_tick:
            obs=np.clip((self.observation(data)-self.mean)/np.sqrt(self.var+1e-8),-10,10)
            self.action=self.policy.predict(obs,deterministic=True)[0]
            self.policy_tick=tick
        super().__call__(data)
