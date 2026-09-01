"""Lab 6 TODO starter：你的第一只 RL Pupper。

学生只需要补三处 TODO：`_compute_reward`、`_get_obs`、`_sample_command`。
其余 PPO 配置、训练循环、渲染管线、画图都已经接好。
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.rl.obs_helpers import base_local_gravity, foot_contact_indicator  # noqa: E402

DEFAULT_XML = str(EXERCISES_DIR / "shared" / "models" / "pupper_v3_floating.xml")

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1",  "leg_back_r_2",  "leg_back_r_3",
    "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
]

FOOT_BODIES = [
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
]

REWARD_WEIGHTS = {
    "vel": 1.5,
    "alive": 0.1,
    "torque": 2e-4,
    "action_rate": 1e-2,
    "ori": 1.0,
    "height": 5.0,
}

KP = 15.0
KD = 0.5
ACTION_SCALE = 0.5
TARGET_HEIGHT = 0.18
MAX_STEPS = 1000


def _solve_stance_q() -> np.ndarray:
    """用 shared/upstream 的 IK 解算站姿关节角。"""
    from shared.upstream import lab_4_mujoco as L
    stance2 = L.GAIT_KEY_POINTS[2]
    q = np.zeros(12, dtype=np.float64)
    for i, leg in enumerate(L.LEG_NAMES):
        target = stance2 + L.LEG_EE_OFFSETS[leg]
        theta, _ = L.inverse_kinematics_single_leg(
            target, leg, initial_guess=(0.0, 0.0, 0.0), max_iter=500, tol=1e-4,
        )
        q[3 * i : 3 * i + 3] = theta
    return q


_STANCE_Q: np.ndarray | None = None


def _get_stance_q() -> np.ndarray:
    global _STANCE_Q
    if _STANCE_Q is None:
        _STANCE_Q = _solve_stance_q()
    return _STANCE_Q.copy()


class PupperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml: str = DEFAULT_XML, dt: float = 0.02, max_steps: int = MAX_STEPS):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        self.max_steps = max_steps
        self.stance_q = _get_stance_q()

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        obs_dim = 3 + 3 + 12 + 12 + 12 + 4 + 3  # = 49
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32,
        )

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._joint_qpos_ids = []
        self._joint_qvel_ids = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            self._joint_qvel_ids.append(int(self.model.jnt_dofadr[jid]))
        self._joint_qpos_ids = np.array(self._joint_qpos_ids, dtype=int)
        self._joint_qvel_ids = np.array(self._joint_qvel_ids, dtype=int)

        self._foot_body_ids = []
        for name in FOOT_BODIES:
            self._foot_body_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            )

        self._setup_pd()

        self.last_action = np.zeros(12, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    def _setup_pd(self):
        self.model.actuator_gainprm[:, 0] = KP
        self.model.actuator_biasprm[:, 1] = -KP
        self.model.actuator_biasprm[:, 2] = -KD

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, TARGET_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._joint_qpos_ids] = self.stance_q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.stance_q
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.cmd = self._sample_command()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        target = self.stance_q + ACTION_SCALE * action
        self.data.ctrl[:] = target
        n_sub = max(1, int(self.dt / self.model.opt.timestep))
        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)

        reward, info = self._compute_reward(action)
        self.last_action = action.copy()
        self.step_count += 1
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # TODO 1: _compute_reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        """计算 6 项加权奖励之和。

        提示：
        - r_vel = exp(-||v_xy - v_xy_cmd||^2 / sigma^2)，sigma = 0.5
        - r_alive = 1.0（每步存活就给）
        - r_torque = -||tau||^2，用 self.data.qfrc_actuator 读力矩
        - r_action_rate = -||a_t - a_{t-1}||^2
        - r_ori = -(roll^2 + pitch^2)，从 self.data.xmat 提取 roll/pitch
        - r_height = -(z - 0.18)^2
        - 最终 reward = sum(REWARD_WEIGHTS[k] * r_k)

        想一想：为什么 r_torque 的系数是 2e-4 而不是 1？
        如果把它调到 1e-2，PPO 探索期会发生什么？
        """
        raise NotImplementedError(
            "TODO 1: 实现 _compute_reward。\n"
            "需要计算 6 项奖励（vel / alive / torque / action_rate / ori / height），\n"
            "按 REWARD_WEIGHTS 加权求和。返回 (float, dict)，dict 里放各项分量。\n"
            "提示：主奖励 r_vel 用 exp(-||error||^2 / 0.25)，不是直接 -||error||^2。"
        )

    # ------------------------------------------------------------------
    # TODO 2: _get_obs
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """拼接 49 维观测向量。

        顺序必须和 observation_space 对齐：
          base_omega(3) + base_local_gravity(3) + qpos(12) + qvel(12)
          + last_action(12) + foot_contact(4) + cmd(3)

        提示：
        - base 系下重力 = R_base.T @ [0, 0, -1]，用 shared.rl.obs_helpers.base_local_gravity
        - foot_contact 判断：data.cfrc_ext[foot_body_id, 2] > 0.5
        - 别忘了 last_action 进 obs，少了它策略会高频抖动
        """
        raise NotImplementedError(
            "TODO 2: 实现 _get_obs。\n"
            "拼接 49 维向量：base_omega(3) + gravity(3) + qpos(12) + qvel(12)\n"
            "+ last_action(12) + foot_contact(4) + cmd(3)。\n"
            "提示：gravity 用 base_local_gravity(model, data, base_id)，\n"
            "foot_contact 用 cfrc_ext[body_id, 2] > 0.5 判断。"
        )

    # ------------------------------------------------------------------
    # TODO 3: _sample_command
    # ------------------------------------------------------------------
    def _sample_command(self) -> np.ndarray:
        """采样一个速度命令 (vx, vy, wz)。

        提示：
        - vx ~ U(-0.4, 0.6)，vy ~ U(-0.2, 0.2)，wz ~ U(-0.6, 0.6)
        - 用 self.np_random.uniform(low, high)
        - 范围太大学不会，太小没意义——这三个范围是教程 §6.2 的经验值
        """
        raise NotImplementedError(
            "TODO 3: 实现 _sample_command。\n"
            "返回 np.array([vx, vy, wz], dtype=np.float32)。\n"
            "提示：vx ~ U(-0.4, 0.6)，vy ~ U(-0.2, 0.2)，wz ~ U(-0.6, 0.6)。\n"
            "用 self.np_random.uniform(low, high) 采样。"
        )

    def _is_fallen(self) -> bool:
        base_z = float(self.data.qpos[2])
        if base_z < 0.10:
            return True
        R = self.data.xmat[self._base_id].reshape(3, 3)
        roll = abs(float(np.arctan2(R[2, 1], R[2, 2])))
        pitch = abs(float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))))
        return roll > 0.7 or pitch > 0.7
