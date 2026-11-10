"""Quaddle RL 环境冒烟测试。

运行：cd codes/practices/quadruped/cs123 && uv run pytest -q 8.rl_quaddle/tests
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def env():
    import sys

    sys.path.insert(0, str(LAB_DIR))
    from quaddle_env import QuaddleEnv

    e = QuaddleEnv(perturb_enabled=True)
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (26,)
    assert env.action_space.shape == (6,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (26,)
    assert np.all(np.isfinite(obs))


def test_random_rollout_stays_finite(env):
    obs, _ = env.reset(seed=1)
    for _ in range(200):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert np.isfinite(rew)
        if term or trunc:
            obs, _ = env.reset()
    assert True


def test_reward_components_ordering():
    """无扰动 + 零命令 + 零动作：100 步应拿满存活+跟踪奖励且不翻倒。"""
    import sys
    sys.path.insert(0, str(LAB_DIR))
    from quaddle_env import QuaddleEnv

    e = QuaddleEnv(perturb_enabled=False)
    obs, _ = e.reset(seed=2)
    e._cmd[:] = [0.0, 0.0, 0.0]
    total = 0.0
    # amp01=-1 → 摆幅 0°（静止站桩）；v2 里零动作=7° 摆动，故显式关摆
    for _ in range(100):
        obs, rew, term, trunc, info = e.step(np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        total += rew
        if term or trunc:
            break
    e.close()
    assert np.isfinite(total)
    # 零命令完美跟踪（≈1.0×2 + 0.5 存活）→ 100 步应远大于 0
    assert total > 200, f"静止 100 步奖励异常低：{total:.1f}"


def test_servo_actuation_moves_joint(env):
    obs, _ = env.reset(seed=3)
    env._cmd[:] = [0.0, 0.0, 0.0]
    q0 = float(env.data.qpos[7])
    for _ in range(100):
        env.step(np.full(6, 1.0))  # 满幅前偏
    q1 = float(env.data.qpos[7])
    assert abs(q1 - q0) > math.radians(3.0), f"舵机未动：Δ={math.degrees(q1 - q0):.1f}°"


def test_termination_on_tip_over(env):
    obs, _ = env.reset(seed=4)
    env._cmd[:] = [0.0, 0.0, 0.0]
    terminated = False
    for _ in range(1000):
        obs, rew, term, trunc, info = env.step(np.full(6, 1.0))
        if term:
            terminated = True
            break
    # 大幅持续偏转应导致翻倒或至少回合结束
    assert terminated or trunc
