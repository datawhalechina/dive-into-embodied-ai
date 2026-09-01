import mujoco
import numpy as np

from pupper_env import PupperEnv, REWARD_WEIGHTS


def test_spaces():
    env = PupperEnv()
    assert env.observation_space.shape == (48,)
    assert env.action_space.shape == (12,)


def test_reset_obs():
    env = PupperEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (48,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)


def test_step_contract():
    env = PupperEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(12, np.float32),
    )
    assert obs.shape == (48,)
    assert np.isfinite(reward)
    assert reward >= 0.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_components_present():
    env = PupperEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    for key in REWARD_WEIGHTS:
        assert f"r_{key}" in info
        assert np.isfinite(info[f"r_{key}"])


def test_determinism_same_seed():
    obs_1, _ = PupperEnv().reset(seed=123)
    obs_2, _ = PupperEnv().reset(seed=123)
    assert np.allclose(obs_1, obs_2)


def test_termination_when_dropped():
    env = PupperEnv()
    env.reset(seed=0)
    env.data.qpos[2] = 0.02
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, _ = env.step(np.zeros(12, np.float32))
    assert terminated is True


def test_termination_when_tilted():
    env = PupperEnv()
    env.reset(seed=0)
    env.data.qpos[2] = 0.25
    half_angle = np.pi / 4
    env.data.qpos[3:7] = [
        np.cos(half_angle), np.sin(half_angle), 0.0, 0.0,
    ]
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, _ = env.step(np.zeros(12, np.float32))
    assert env.data.qpos[2] > 0.10
    assert terminated is True


def test_truncation_at_max_steps():
    env = PupperEnv(max_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.zeros(12, np.float32))
        if terminated:
            break
    assert truncated is True
