"""Pupper RL Gymnasium 环境：obs / action / step / reset / reward。

教程 Ch6 §6.2–§6.4 的端到端实现。12 维连续动作（PD 残差），
49 维观测，6 项加权奖励。
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

EXERCISES_DIR = Path(__file__).resolve().parents[2]
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

    def _get_obs(self) -> np.ndarray:
        base_omega = self.data.qvel[3:6].copy()
        gravity = base_local_gravity(self.model, self.data, self._base_id)
        qpos = self.data.qpos[self._joint_qpos_ids].copy()
        qvel = self.data.qvel[self._joint_qvel_ids].copy()
        foot_contact = foot_contact_indicator(self.model, self.data, self._foot_body_ids)
        obs = np.concatenate([
            base_omega,
            gravity,
            qpos,
            qvel,
            self.last_action,
            foot_contact,
            self.cmd,
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        base_vel = self.data.qvel[0:3]
        vx, vy = float(base_vel[0]), float(base_vel[1])
        vx_cmd, vy_cmd = float(self.cmd[0]), float(self.cmd[1])

        r_vel = float(np.exp(-((vx - vx_cmd) ** 2 + (vy - vy_cmd) ** 2) / 0.25))
        r_alive = 1.0

        ctrl_torque = self.data.qfrc_actuator[self._joint_qvel_ids]
        r_torque = -float(np.sum(ctrl_torque ** 2))

        r_action_rate = -float(np.sum((action - self.last_action) ** 2))

        quat = self.data.qpos[3:7]
        R = self.data.xmat[self._base_id].reshape(3, 3)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
        r_ori = -(roll ** 2 + pitch ** 2)

        base_z = float(self.data.qpos[2])
        r_height = -((base_z - TARGET_HEIGHT) ** 2)

        w = REWARD_WEIGHTS
        reward = (
            w["vel"] * r_vel
            + w["alive"] * r_alive
            + w["torque"] * r_torque
            + w["action_rate"] * r_action_rate
            + w["ori"] * r_ori
            + w["height"] * r_height
        )
        info = {
            "r_vel": r_vel, "r_alive": r_alive, "r_torque": r_torque,
            "r_action_rate": r_action_rate, "r_ori": r_ori, "r_height": r_height,
        }
        return float(reward), info

    def _sample_command(self) -> np.ndarray:
        vx = self.np_random.uniform(-0.4, 0.6)
        vy = self.np_random.uniform(-0.2, 0.2)
        wz = self.np_random.uniform(-0.6, 0.6)
        return np.array([vx, vy, wz], dtype=np.float32)

    def _is_fallen(self) -> bool:
        base_z = float(self.data.qpos[2])
        if base_z < 0.10:
            return True
        R = self.data.xmat[self._base_id].reshape(3, 3)
        roll = abs(float(np.arctan2(R[2, 1], R[2, 2])))
        pitch = abs(float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))))
        return roll > 0.7 or pitch > 0.7
