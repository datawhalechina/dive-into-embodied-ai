"""Terrain residual PPO, using privileged terrain probes and simulator velocity.

Actions modify only four actuator targets. Startup and 1 ms physics / 5 ms
forward passes match scenario_matrix. These observations require estimation on hardware.
"""
import math
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from gait_controller import JointController
from scenario_matrix import profile
from scenarios import load_scene, make_scene
from run_sim import SIDES


class ResidualController:
    def __init__(self, model, scene, rotation, config, scale=.25):
        self.model, self.scene, self.rotation, self.config = model, scene, rotation, config
        self.base = JointController('walk', config=config)
        self.scale = scale
        self.action = np.zeros(4)
        self.reference = np.zeros(4)
        self.target = np.zeros(4)
        self.tick = -1
        self.qids = [model.jnt_qposadr[model.joint(s+'_thigh').id] for s in SIDES]
        self.vids = [model.jnt_dofadr[model.joint(s+'_thigh').id] for s in SIDES]

    def observation(self, data):
        rot = data.body('body').xmat.reshape(3, 3)
        relative = self.rotation.T @ rot
        yaw = math.atan2(relative[1, 0], relative[0, 0])
        phase = 2*math.pi*self.config['frequency_hz']*max(0., data.time-2.)
        x,y,z = data.qpos[:3]
        heights = [(self.scene.surface_height(x, y-d)-z)/.05 for d in [0., .025, .05, .075, .1]]
        return np.concatenate((rot.T@np.array([0.,0.,-1.]), rot.T@data.qvel[:3]/.1,
            data.qvel[3:6]/5., data.qpos[self.qids], data.qvel[self.vids]/10.,
            self.action, self.reference, [math.sin(phase),math.cos(phase),.75,x/.1,math.sin(yaw),math.cos(yaw)], heights)).astype(np.float32)

    def __call__(self, data):
        self.base(data)
        self.reference = data.ctrl.copy()
        if data.time < 2.-1e-8:
            self.target = self.reference.copy()
            return
        tick = round(data.time/.01)
        if abs(data.time-tick*.01)<1e-7 and tick != self.tick:
            self.tick = tick
            desired = np.clip(self.reference+self.scale*self.action, -1.2, 1.2)
            self.target += np.clip(desired-self.target, -.08, .08)
        data.ctrl[:] = self.target


class TerrainEnv(gym.Env):
    metadata = {'render_modes': []}
    controller_class = ResidualController
    def __init__(self, scene='stairs_up', seconds=20., seed_noise=False, scene_definition=None):
        if scene not in ('flat', 'uphill3', 'uphill5', 'downhill3', 'stairs_up', 'stairs_down', 'rough'):
            raise ValueError('Terrain PPO currently supports dry locomotion scenes only')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('seconds must be finite and positive')
        self.scene = scene_definition if scene_definition is not None else make_scene(scene)
        self.model, self.data, self.rotation = load_scene(self.scene)
        if not math.isclose(self.model.opt.timestep, .001):
            raise ValueError('This rollout interface requires a 1 ms physics step')
        self.initial = self.data.qpos.copy()
        self.config = profile('walk')
        self.seconds, self.seed_noise = seconds, seed_noise
        self.observation_space = spaces.Box(-np.inf,np.inf,(36,),np.float32)
        self.action_space = spaces.Box(-1.,1.,(4,),np.float32)
        self.body_geom = self.model.geom('body_contact').id
        self.axis = self.rotation@np.array([0.,-1.,0.])

    def physics(self):
        self.controller(self.data)
        mujoco.mj_step(self.model,self.data)
        if self.physics_steps%5==0:
            mujoco.mj_forward(self.model,self.data)
        self.physics_steps+=1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=self.initial
        if self.seed_noise:
            self.data.qpos[0] += self.np_random.uniform(-.005,.005)
        mujoco.mj_forward(self.model,self.data)
        self.controller=self.controller_class(self.model,self.scene,self.rotation,self.config)
        self.physics_steps=0
        for _ in range(2000): self.physics()
        self.start=self.data.qpos[:3].copy()
        self.steps=0
        self.done=False
        return self.controller.observation(self.data), {}

    def step(self, action):
        if self.done: raise RuntimeError('reset required')
        action=np.asarray(action,dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all(): raise ValueError('four finite actions required')
        previous=self.controller.action.copy()
        self.controller.action=np.clip(action,-1,1)
        before=self.data.qpos[:3].copy()
        contact=False; peak_speed=0.
        for _ in range(10):
            self.physics()
            contact |= any(self.body_geom in (c.geom1,c.geom2) for c in self.data.contact)
            peak_speed=max(peak_speed,float(np.max(np.abs(self.data.qvel[self.controller.vids]))))
        self.steps+=1
        rot=self.rotation.T@self.data.body('body').xmat.reshape(3,3)
        yaw=math.atan2(rot[1,0],rot[0,0])
        upright=float(rot[2,2]); clearance=float(self.data.qpos[2]-self.scene.surface_height(*self.data.qpos[:2]))
        distance=float((self.data.qpos[:3]-self.start)@self.axis)
        speed=float((self.data.qpos[:3]-before)@self.axis/.01)
        fallen=contact or upright<.8 or clearance<.025 or not np.isfinite(self.data.qpos).all()
        # Dense progress with explicit penalties for the same evaluation failures.
        reward=.01*(2*np.clip(speed/.075,-2,2)+math.exp(-((speed-.075)/.07)**2)
            -.8*(yaw/.436)**2-.2*(self.data.qpos[0]/.1)**2
            -.4*max(0.,peak_speed-9.)**2-.05*np.mean(action**2)
            -.1*np.mean((action-previous)**2))-(2. if fallen else 0.)
        terminated=bool(fallen)
        truncated=self.steps>=round(self.seconds*100) and not terminated
        self.done=terminated or truncated
        return self.controller.observation(self.data),float(reward),terminated,truncated,{
            'forward_m':distance,'fallen':terminated,'peak_joint_speed':peak_speed,'yaw_deg':math.degrees(yaw)}


class PolicyController(ResidualController):
    def __init__(self, model, scene, rotation, config, directory):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        import json
        super().__init__(model,scene,rotation,config)
        directory=Path(directory)
        self.policy=PPO.load(directory/'policy.zip',device='cpu')
        self.policy_tick=-1
        with np.load(directory/'obs_stats.npz') as stats:
            self.mean, self.var = stats['mean'].copy(), stats['var'].copy()
        metadata=json.loads((directory/'metadata.json').read_text())
        self.scale=metadata['residual_scale_rad']

    def __call__(self,data):
        tick=round(data.time/.01)
        if data.time>=2.-1e-8 and abs(data.time-tick*.01)<1e-7 and tick!=self.policy_tick:
            obs=self.observation(data)
            obs=np.clip((obs-self.mean)/np.sqrt(self.var+1e-8),-10,10)
            self.action=self.policy.predict(obs,deterministic=True)[0]
            self.policy_tick=tick
        super().__call__(data)
