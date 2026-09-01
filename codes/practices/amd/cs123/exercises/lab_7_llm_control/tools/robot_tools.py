"""Lab 7 工具定义 + dispatch 实现。

TOOLS 列表供 OpenAI-compatible SDK function calling 使用；
dispatch() 执行工具并返回 JSON-able 结果。

物理后端复用 lab5 的 RTNeural policy（默认 test，和 lab6 一致），
跑 50 Hz policy / 500 Hz physics 的同一套循环。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------

_TOOLS_DIR = Path(__file__).resolve().parent
LAB_DIR = _TOOLS_DIR.parent
EXERCISES_DIR = LAB_DIR.parent
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.upstream.lab_5_mujoco import (  # noqa: E402
    RTNeuralPolicy,
    build_obs_frame,
    load_policy,
    quat_inv_rotate,
    roll_obs_history,
    OBS_TOTAL,
)

MODEL_FLOATING = EXERCISES_DIR / "shared" / "models" / "pupper_v3_floating.xml"
PHYS_DT = 0.002
CONTROL_DT = 0.02
CONTROL_RATIO = int(round(CONTROL_DT / PHYS_DT))

# ---------------------------------------------------------------------------
# 工具 JSON schema（教程 §7.2）
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "walk",
            "description": (
                "Walk in the body frame for a fixed duration. "
                "Stops automatically when the duration elapses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {
                        "type": "number",
                        "description": "forward speed m/s, range [-1.5, 1.5]. Values below 0.2 may not produce visible motion.",
                    },
                    "wz": {
                        "type": "number",
                        "description": "yaw rate rad/s, range [-2.0, 2.0]",
                    },
                    "duration": {
                        "type": "number",
                        "description": "seconds, range (0, 10]",
                    },
                },
                "required": ["vx", "wz", "duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Immediately set commanded velocity to 0.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pose",
            "description": "Return current (x, y, yaw) in world frame.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Hold position for N seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "seconds, range (0, 10]"},
                },
                "required": ["seconds"],
            },
        },
    },
]

VX_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)
DURATION_RANGE = (0.0, 10.0)


# ---------------------------------------------------------------------------
# RobotState — 封装 MuJoCo model/data + RTNeural policy 的运行状态
# ---------------------------------------------------------------------------

class RobotState:
    """和 lab6 的 KarelPupperSim 角色等价，但面向 offscreen 渲染（无 viewer）。

    内部维护 obs_history / last_action，提供 step_cmd() 让 dispatch 驱动。
    """

    def __init__(self, policy_name: str = "test"):
        self.policy = load_policy(policy_name)

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_FLOATING))
        self.model.opt.timestep = PHYS_DT
        self.model.actuator_gainprm[:, 0] = self.policy.kp
        self.model.actuator_biasprm[:, 1] = -self.policy.kp
        self.model.actuator_biasprm[:, 2] = -self.policy.kd
        self.data = mujoco.MjData(self.model)

        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link",
        )

        self.obs_history = np.zeros(OBS_TOTAL, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_joints = self.policy.default_joint_pos.copy()
        self.phys_step = 0

        self._reset_home()

    def _reset_home(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.22]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:19] = self.policy.default_joint_pos
        self.data.ctrl[:] = self.policy.default_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.obs_history[:] = 0.0
        self.last_action[:] = 0.0
        self.target_joints = self.policy.default_joint_pos.copy()
        self.phys_step = 0

    def reset(self) -> None:
        self._reset_home()
        self._settle(0.5)

    def _settle(self, seconds: float = 0.5) -> None:
        """PD 静稳：零 action 跑一段让机体站稳（和 lab5/lab6 的 settle 一致）。

        settle 期间 action=0（不跑 policy.forward），但构建 obs_history。
        """
        n = int(seconds / PHYS_DT)
        cmd = np.zeros(3, dtype=np.float32)
        desired_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for s in range(n):
            if s % CONTROL_RATIO == 0:
                quat = self.data.xquat[self.base_id].copy()
                world_ang = self.data.cvel[self.base_id, 0:3].copy()
                local_ang = quat_inv_rotate(quat, world_ang)
                gravity_b = quat_inv_rotate(quat, np.array([0.0, 0.0, -1.0]))
                n_g = np.linalg.norm(gravity_b)
                if n_g > 1e-6:
                    gravity_b = gravity_b / n_g
                joint_pos = self.data.qpos[7:19].copy()

                frame = build_obs_frame(
                    local_ang_vel=local_ang,
                    gravity_in_body=gravity_b,
                    command=cmd,
                    desired_world_z_in_body=desired_z,
                    joint_pos=joint_pos,
                    last_action=self.last_action,
                    default_pose=self.policy.default_joint_pos.astype(np.float64),
                )
                self.obs_history = roll_obs_history(self.obs_history, frame)
                action = np.zeros(12, dtype=np.float32)
                self.last_action = action.copy()
                self.target_joints = (
                    self.policy.default_joint_pos + self.policy.action_scale * action
                )
                self.data.ctrl[:] = self.target_joints

            mujoco.mj_step(self.model, self.data)
            self.phys_step += 1

    def _policy_tick(self, cmd: np.ndarray) -> None:
        """50 Hz policy 更新（和 lab5 _run_policy 里的逻辑一致）。"""
        quat = self.data.xquat[self.base_id].copy()
        world_ang = self.data.cvel[self.base_id, 0:3].copy()
        local_ang = quat_inv_rotate(quat, world_ang)
        gravity_b = quat_inv_rotate(quat, np.array([0.0, 0.0, -1.0]))
        n = np.linalg.norm(gravity_b)
        if n > 1e-6:
            gravity_b = gravity_b / n
        joint_pos = self.data.qpos[7:19].copy()
        desired_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        frame = build_obs_frame(
            local_ang_vel=local_ang,
            gravity_in_body=gravity_b,
            command=cmd,
            desired_world_z_in_body=desired_z,
            joint_pos=joint_pos,
            last_action=self.last_action,
            default_pose=self.policy.default_joint_pos.astype(np.float64),
        )
        self.obs_history = roll_obs_history(self.obs_history, frame)
        action = self.policy.forward(self.obs_history)
        self.last_action = action.copy()
        self.target_joints = (
            self.policy.default_joint_pos + self.policy.action_scale * action
        )

    def step_cmd(
        self,
        vx: float,
        wz: float,
        duration: float,
        *,
        frame_callback: Callable[["RobotState"], None] | None = None,
    ) -> dict[str, Any]:
        """以 (vx, 0, wz) 命令跑 duration 秒，返回结果。

        frame_callback(self) 在每个物理步后被调用，用于 GIF 采帧。
        """
        cmd = np.array([vx, 0.0, wz], dtype=np.float32)
        n_phys = max(1, int(duration / PHYS_DT))

        for s in range(n_phys):
            if s % CONTROL_RATIO == 0:
                self._policy_tick(cmd)
            self.data.ctrl[:] = self.target_joints
            mujoco.mj_step(self.model, self.data)
            self.phys_step += 1

            if frame_callback is not None:
                frame_callback(self)

            if self.is_fallen():
                return {
                    "ok": False,
                    "error": "robot fell mid-walk",
                    "final_pose": self.get_pose(),
                }

        return {"ok": True, "final_pose": self.get_pose()}

    def get_pose(self) -> list[float]:
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        quat = self.data.xquat[self.base_id].copy()
        yaw = float(np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        ))
        return [round(x, 3), round(y, 3), round(yaw, 3)]

    def is_fallen(self) -> bool:
        base_z = float(self.data.qpos[2])
        if base_z < 0.05:
            return True
        R = self.data.xmat[self.base_id].reshape(3, 3)
        roll = abs(float(np.arctan2(R[2, 1], R[2, 2])))
        pitch = abs(float(np.arctan2(
            -R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2),
        )))
        return roll > 0.7 or pitch > 0.7

    @property
    def sim_time(self) -> float:
        return self.phys_step * PHYS_DT


# ---------------------------------------------------------------------------
# dispatch 入口
# ---------------------------------------------------------------------------

def dispatch(
    name: str,
    args: dict[str, Any],
    robot: RobotState,
    *,
    frame_callback: Callable[[RobotState], None] | None = None,
) -> dict[str, Any]:
    """执行工具调用，返回 JSON-able 结果。绝不 raise，只在返回值里写错误。"""
    try:
        if name == "walk":
            vx = float(args.get("vx", 0.0))
            wz = float(args.get("wz", 0.0))
            duration = float(args.get("duration", 1.0))
            if not (VX_RANGE[0] <= vx <= VX_RANGE[1]):
                return {"ok": False, "error": f"vx out of range [{VX_RANGE[0]}, {VX_RANGE[1]}]"}
            if not (WZ_RANGE[0] <= wz <= WZ_RANGE[1]):
                return {"ok": False, "error": f"wz out of range [{WZ_RANGE[0]}, {WZ_RANGE[1]}]"}
            if not (DURATION_RANGE[0] < duration <= DURATION_RANGE[1]):
                return {"ok": False, "error": f"duration out of range (0, {DURATION_RANGE[1]}]"}
            if robot.is_fallen():
                return {"ok": False, "error": "robot has fallen, cannot walk"}
            return robot.step_cmd(vx, wz, duration, frame_callback=frame_callback)

        elif name == "stop":
            return robot.step_cmd(0.0, 0.0, 0.5, frame_callback=frame_callback)

        elif name == "get_pose":
            return {"ok": True, "pose": robot.get_pose()}

        elif name == "wait":
            seconds = float(args.get("seconds", 1.0))
            if not (DURATION_RANGE[0] < seconds <= DURATION_RANGE[1]):
                return {"ok": False, "error": f"seconds out of range (0, {DURATION_RANGE[1]}]"}
            return robot.step_cmd(0.0, 0.0, seconds, frame_callback=frame_callback)

        else:
            return {"ok": False, "error": f"unknown tool: {name}"}

    except Exception as e:
        return {"ok": False, "error": str(e)}
