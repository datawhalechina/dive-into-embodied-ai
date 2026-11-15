"""Optional reset-only terrain sampling and easier-level replay for curriculum runs."""
from collections import OrderedDict
import gymnasium as gym
import numpy as np
from curriculum_rl import CurriculumEnv,curriculum_scene
from scenarios import load_scene


class MixedCurriculumEnv(CurriculumEnv):
    def __init__(self,scene='rough',level=1,terrain_seed=7,seed_noise=True,seconds=20.,
                 terrain_seeds=(7,11,13,17),easy_probability=.2):
        if not terrain_seeds or not 0 <= easy_probability < 1:
            raise ValueError('Nonempty training seeds and probability in [0,1) required')
        super().__init__(scene,level,terrain_seed,seed_noise,seconds)
        self.task_name=scene;self.max_level=level
        self.terrain_seeds=tuple(terrain_seeds);self.easy_probability=easy_probability
        self.cache=OrderedDict()
        self.cache[(level,terrain_seed)]=(self.scene,self.model,self.data,self.rotation,self.initial.copy())

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        level=self.max_level
        if level>1 and self.np_random.random()<self.easy_probability:
            level=int(self.np_random.integers(1,level))
        terrain_seed=int(self.np_random.choice(self.terrain_seeds)) if self.task_name=='rough' else 7
        key=(level,terrain_seed)
        if key not in self.cache:
            scene=curriculum_scene(self.task_name,level,terrain_seed)
            model,data,rotation=load_scene(scene)
            self.cache[key]=(scene,model,data,rotation,data.qpos.copy())
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        self.scene,self.model,self.data,self.rotation,self.initial=self.cache[key]
        self.body_geom=self.model.geom('body_contact').id
        self.axis=self.rotation@np.array([0.,-1.,0.])
        self.active_level=level;self.active_terrain_seed=terrain_seed
        obs,info=super().reset(seed=None,options=options)
        info.update(level=level,terrain_seed=terrain_seed)
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        info.update(level=self.active_level,terrain_seed=self.active_terrain_seed)
        return obs,reward,terminated,truncated,info
