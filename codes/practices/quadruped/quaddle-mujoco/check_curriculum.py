"""Verify terrain identity, probe rotation and repeatable physical rollouts."""
import math
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
from curriculum_rl import CurriculumEnv,curriculum_scene,probe_points,GridController
from scenarios import make_scene,load_scene
from scenario_matrix import profile

for name in ['rough','stairs_up']:
    actual=curriculum_scene(name,3)
    expected=make_scene(name)
    assert actual.boxes==expected.boxes and actual.plane_z==expected.plane_z
    env=CurriculumEnv(name,level=1,seconds=.05)
    check_env(env)
    first,_=env.reset(seed=123);second,_=env.reset(seed=123)
    np.testing.assert_array_equal(first,second)
    assert len(first)==60
    scene=curriculum_scene(name,1);model,data,rotation=load_scene(scene)
    # Same random initial translation and same physics scheduling as the env.
    data.qpos[0]+=np.random.default_rng(123).uniform(-.005,.005)
    mujoco.mj_forward(model,data)
    controller=GridController(model,scene,rotation,profile('walk'))
    for step in range(2000):
        controller(data);mujoco.mj_step(model,data)
        if step%5==0:mujoco.mj_forward(model,data)
    np.testing.assert_allclose(controller.observation(data),second,rtol=0,atol=1e-12)
    for tick in range(5):
        action=np.sin(np.arange(4)+tick)*.2;controller.action=action
        obs,reward,done,truncated,_=env.step(action)
        for step in range(2000+tick*10,2010+tick*10):
            controller(data);mujoco.mj_step(model,data)
            if step%5==0:mujoco.mj_forward(model,data)
        np.testing.assert_allclose(controller.observation(data),obs,rtol=0,atol=1e-12)
        assert np.isfinite(reward) and not done and truncated==(tick==4)
        assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
    points=probe_points(data)-data.qpos[:2]
    data.qpos[3:7]=[math.sqrt(.5),0,0,math.sqrt(.5)];mujoco.mj_forward(model,data)
    rotated=probe_points(data)-data.qpos[:2]
    np.testing.assert_allclose(rotated,points@np.array([[0,1],[-1,0]]),atol=1e-12)
    print(name,'full difficulty identity / reset / observation-rollout parity / probe rotation PASS')
