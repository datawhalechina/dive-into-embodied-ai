import mujoco
import numpy as np

from pupper_env import (
    GAIT_NAMES,
    PupperEnv,
    REWARD_WEIGHTS,
    TERMINAL_BODY_Z,
)


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
    assert env.data.qpos[2] > TERMINAL_BODY_Z
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


def test_gait_features_are_appended_without_changing_base_observation():
    base_env = PupperEnv()
    gait_env = PupperEnv(gait_enabled=True)
    base_obs, _ = base_env.reset(seed=7)
    gait_obs, _ = gait_env.reset(seed=7)

    assert gait_env.observation_space.shape == (53,)
    assert gait_obs.shape == (53,)
    assert np.allclose(base_obs, gait_obs[:48])
    assert np.isclose(np.sum(gait_obs[48:51]), 1.0)
    assert np.isclose(np.linalg.norm(gait_obs[51:53]), 1.0)


def test_gait_contact_reward_and_diagnostics_present():
    env = PupperEnv(gait_enabled=True, gait_switch_steps=0)
    env.reset(seed=0)
    env.set_gait("trot")
    _, _, _, _, info = env.step(np.zeros(12, np.float32))

    assert info["gait_name"] == "trot"
    assert "r_gait_contact" in info
    assert 0.0 <= info["gait_contact_match"] <= 1.0
    assert len(info["expected_contacts"]) == 4
    assert len(info["actual_contacts"]) == 4


def test_expected_contact_patterns_at_phase_zero():
    env = PupperEnv(gait_enabled=True, gait_switch_steps=0)
    env.reset(seed=0)
    expected = {
        "walk": [True, True, True, True],
        "trot": [True, True, True, True],
        "pace": [True, True, True, True],
    }
    # phase=0 位于边界；向周期内推进一点，检查每种步态的支撑足组合。
    for gait_name in GAIT_NAMES:
        env.set_gait(gait_name)
        env.gait_phase = 0.01
        pattern = env._expected_contacts().tolist()
        if gait_name == "walk":
            expected[gait_name] = [True, False, True, True]
        elif gait_name == "trot":
            expected[gait_name] = [False, True, True, False]
        else:
            expected[gait_name] = [False, True, False, True]
        assert pattern == expected[gait_name]


def test_perturb_disabled_gives_zero_initial_velocity():
    env = PupperEnv(perturb_enabled=False)
    env.reset(seed=0)
    assert np.allclose(env.data.qvel[0:6], 0.0)


def test_perturb_enabled_randomizes_initial_velocity():
    env = PupperEnv(perturb_enabled=True)
    env.reset(seed=0)
    assert np.any(np.abs(env.data.qvel[0:2]) > 1e-6)


def test_command_resample_mid_episode():
    env = PupperEnv(cmd_resample_steps=2, max_steps=10)
    env.reset(seed=0)
    env.cmd = np.array([9.0, 9.0, 9.0], dtype=np.float32)
    env.step(np.zeros(12, np.float32))
    assert np.allclose(env.cmd, [9.0, 9.0, 9.0])
    env.step(np.zeros(12, np.float32))
    assert not np.allclose(env.cmd, [9.0, 9.0, 9.0])


def test_command_resample_disabled():
    env = PupperEnv(cmd_resample_steps=0, max_steps=10)
    env.reset(seed=0)
    env.cmd = np.array([9.0, 9.0, 9.0], dtype=np.float32)
    for _ in range(5):
        env.step(np.zeros(12, np.float32))
    assert np.allclose(env.cmd, [9.0, 9.0, 9.0])


def test_gait_switch_keeps_episode_step_count():
    env = PupperEnv(
        gait_enabled=True,
        gait_types=("walk", "trot"),
        gait_switch_steps=1,
        max_steps=3,
    )
    env.reset(seed=0)
    first_gait = env.gait_name
    env.step(np.zeros(12, np.float32))
    assert env.gait_name != first_gait
    assert env.gait_phase == 0.0
    assert env.step_count == 1
