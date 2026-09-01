"""Lab 6 数值断言（不依赖训练完成的 checkpoint，跑 < 30 s）。

env 已对齐 MJX/brax 配方：540 维 obs（36 × 15 stack），18 项 reward，宽命令分布。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.pupper_env import (  # noqa: E402
    CMD_VX_RANGE, CMD_VY_RANGE, CMD_WZ_RANGE,
    OBS_DIM_STACKED, PupperEnv, REWARD_WEIGHTS,
)

EXPECTED_REWARD_KEYS = {
    "tracking_lin_vel", "tracking_ang_vel", "tracking_orientation",
    "lin_vel_z", "ang_vel_xy", "orientation",
    "torques", "joint_acceleration", "mechanical_work",
    "action_rate", "feet_air_time",
    "stand_still", "stand_still_joint_velocity", "abduction_angle",
    "termination", "foot_slip", "knee_collision", "body_collision",
}


def test_spaces():
    env = PupperEnv()
    assert env.observation_space.shape == (OBS_DIM_STACKED,), \
        f"obs shape 不对: {env.observation_space.shape}"
    assert env.action_space.shape == (12,), f"action shape 不对: {env.action_space.shape}"
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "reset 后 obs 含有 inf/nan"
    print("  test_spaces 通过")


def test_reset_obs():
    env = PupperEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM_STACKED,), f"reset obs shape: {obs.shape}"
    assert obs.dtype == np.float32, f"reset obs dtype: {obs.dtype}"
    # obs 前 6 维是 lagged IMU = (ang_vel 3, gravity 3)；初始化时 buffer 全 -1 z 分量
    gravity = obs[3:6]
    assert np.abs(np.linalg.norm(gravity) - 1.0) < 0.5, \
        f"重力向量模应接近 1（含噪声允许 ±0.5）: {gravity}"
    print("  test_reset_obs 通过")


def test_step_rewards():
    env = PupperEnv()
    env.reset(seed=0)
    action = np.zeros(12, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    # 所有 18 项都应有 r_* key（值已乘 weight，可正可负或 0）
    for k in EXPECTED_REWARD_KEYS:
        assert f"r_{k}" in info, f"info 缺 r_{k}"
    # tracking 项 weight 正、term 正 → 贡献 >= 0
    assert info["r_tracking_lin_vel"] >= 0
    assert info["r_tracking_ang_vel"] >= 0
    assert info["r_tracking_orientation"] >= 0
    # 几个 penalty 项 weight 负、term 正 → 贡献 <= 0
    assert info["r_torques"] <= 1e-6, f"r_torques 应 <= 0: {info['r_torques']}"
    assert info["r_action_rate"] <= 1e-6, f"r_action_rate 应 <= 0: {info['r_action_rate']}"
    assert np.isfinite(reward), f"总 reward 不是 finite: {reward}"
    # reward 已 clip 到 [0, 10000]
    assert 0.0 <= reward <= 10000.0, f"reward 超出 clip 范围: {reward}"
    print("  test_step_rewards 通过")


def test_reward_weights_and_command():
    assert EXPECTED_REWARD_KEYS == set(REWARD_WEIGHTS.keys()), \
        f"REWARD_WEIGHTS keys 与预期 18 项不一致: {set(REWARD_WEIGHTS.keys()) ^ EXPECTED_REWARD_KEYS}"
    for k, v in REWARD_WEIGHTS.items():
        assert isinstance(v, (int, float)), f"REWARD_WEIGHTS[{k!r}] 应为数字: {v}"

    env = PupperEnv()
    env.reset(seed=0)
    vx_samples, vy_samples, wz_samples = [], [], []
    for _ in range(200):
        cmd = env._sample_command()
        vx_samples.append(cmd[0])
        vy_samples.append(cmd[1])
        wz_samples.append(cmd[2])
    vx_arr = np.array(vx_samples)
    vy_arr = np.array(vy_samples)
    wz_arr = np.array(wz_samples)
    # zero-cmd-prob 1% 会让边界稍稍向 0 拉，但绝大多数仍在区间内
    assert np.all((CMD_VX_RANGE[0] <= vx_arr) & (vx_arr <= CMD_VX_RANGE[1])), \
        f"vx 超出 {CMD_VX_RANGE}: min/max = {vx_arr.min()}/{vx_arr.max()}"
    assert np.all((CMD_VY_RANGE[0] <= vy_arr) & (vy_arr <= CMD_VY_RANGE[1])), \
        f"vy 超出 {CMD_VY_RANGE}"
    assert np.all((CMD_WZ_RANGE[0] <= wz_arr) & (wz_arr <= CMD_WZ_RANGE[1])), \
        f"wz 超出 {CMD_WZ_RANGE}"
    print("  test_reward_weights_and_command 通过")


def main() -> None:
    print("Lab 6 数值断言:")
    test_spaces()
    test_reset_obs()
    test_step_rewards()
    test_reward_weights_and_command()
    print("Lab 6 检查全部通过。")


if __name__ == "__main__":
    main()
