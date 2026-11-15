"""Physical startup and residual rollout parity; no mock dynamics."""
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
from terrain_rl import TerrainEnv, ResidualController
from scenarios import make_scene,load_scene
from scenario_matrix import profile


def main():
    check_env(TerrainEnv('flat',seconds=.05))
    for name in ['flat','uphill5','stairs_up','stairs_down','rough']:
        env=TerrainEnv(name,seconds=.1); first,_=env.reset(seed=123)
        second,_=env.reset(seed=123);np.testing.assert_array_equal(first,second)
        scene=make_scene(name);model,data,rotation=load_scene(scene)
        controller=ResidualController(model,scene,rotation,profile('walk'))
        for step in range(2000):
            controller(data);mujoco.mj_step(model,data)
            if step%5==0:mujoco.mj_forward(model,data)
        np.testing.assert_allclose(env.data.qpos,data.qpos,rtol=0,atol=1e-12)
        for tick in range(10):
            action=np.sin(np.arange(4)+tick)*.2
            controller.action=action
            obs,reward,done,truncated,_=env.step(action)
            for step in range(2000+tick*10,2010+tick*10):
                controller(data);mujoco.mj_step(model,data)
                if step%5==0:mujoco.mj_forward(model,data)
            np.testing.assert_allclose(env.data.qpos,data.qpos,rtol=0,atol=1e-12)
            assert np.isfinite(reward) and not done
            assert truncated==(tick==9)
            assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        print(name,'reset/rollout parity, joint-only and timeout PASS',flush=True)

if __name__=='__main__':main()
