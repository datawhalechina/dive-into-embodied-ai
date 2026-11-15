"""Check terrain identity, reproducibility and absence of direct physical assistance."""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from practice_rl import PracticeEnv,practice_scene
from scenarios import make_scene
from curriculum_rl import CurriculumEnv

for name in ['rough','stairs_up']:
    a,b=practice_scene(name),make_scene(name)
    assert a.boxes==b.boxes and a.plane_z==b.plane_z
    env=PracticeEnv(name,seconds=.02)
    check_env(env)
    a,ia=env.reset(seed=123);b,ib=env.reset(seed=123)
    np.testing.assert_array_equal(a,b);assert ia==ib
    for seed in range(5):
        obs,info=env.reset(seed=seed)
        assert len(env.cache)<=2 and abs(info['yaw_rad'])<=np.deg2rad(3)
        assert .9<=info['friction_scale']<=1.1
        current=env.scene.metadata();friction=env.model.geom_friction.copy()
        for tick in range(2):
            obs,reward,terminated,truncated,result=env.step(np.zeros(4))
            assert np.isfinite(reward) and not terminated and truncated==(tick==1)
            assert env.scene.metadata()==current
            np.testing.assert_array_equal(env.model.geom_friction,friction)
            assert not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
    env.close()
    print(name,'reset / full terrain identity / fixed episode physics / joint-only PASS')
# Default gain must preserve old rollouts; positive gain modifies targets, not qvel.
a=CurriculumEnv('stairs_up',seconds=.01);b=CurriculumEnv('stairs_up',seconds=.01)
b.config=dict(b.config,speed_feedback_gain=0.)
a.reset(seed=1);b.reset(seed=1)
a.step(np.zeros(4));b.step(np.zeros(4))
np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
a.close();b.close()
print('zero feedback legacy parity PASS')
