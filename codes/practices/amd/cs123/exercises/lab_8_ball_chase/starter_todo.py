"""Lab 8 学生版：追球 Demo（Ball-Chasing Demo）。

补全 3 处 TODO，让 Pupper 看到红球、走过去、在球前停下。

物理后端严格复用 Lab 7 的 RobotState（test_policy.json + 50 Hz / 500 Hz），
只替换 XML 路径为 pupper_chase.xml（加了 head_cam + mocap 红球）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.viz import gif_utils  # noqa: E402

LAB_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = LAB_DIR / "portfolio"
MODEL_CHASE = LAB_DIR / "models" / "pupper_chase.xml"

_LAB7_DIR = EXERCISES_DIR / "lab_7_llm_control"
if str(_LAB7_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB7_DIR))
if str(_LAB7_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(_LAB7_DIR / "tools"))

from robot_tools import RobotState, PHYS_DT, CONTROL_DT, CONTROL_RATIO  # noqa: E402

GIF_FPS = 12
GIF_WIDTH = 640
GIF_HEIGHT = 480
VISION_HZ = 10
CHASE_DURATION = 20.0
BALL_MOVE_INTERVAL = 3.0

# P 控制器参数
KP_YAW = 1.0
KP_VX = 0.4
H_REF = 30  # 像素
SEARCH_WZ = 0.6
LOST_TIMEOUT = 2.0

# HSV 阈值
_LOW1 = np.array([0, 120, 80], dtype=np.uint8)
_HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
_LOW2 = np.array([170, 120, 80], dtype=np.uint8)
_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)
_MIN_AREA = 50


# ---------------------------------------------------------------------------
# ChaseRobotState — 只替换 XML 路径，其余完全继承 Lab 7
# ---------------------------------------------------------------------------


class ChaseRobotState(RobotState):
    """Lab 7 RobotState 的子类，唯一区别是加载 pupper_chase.xml。"""

    def __init__(self, policy_name: str = "test"):
        from shared.upstream.lab_5_mujoco import load_policy, OBS_TOTAL  # noqa: F811

        self.policy = load_policy(policy_name)

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_CHASE))
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
        self.data.mocap_pos[0] = [1.5, 0.3, 0.05]
        mujoco.mj_forward(self.model, self.data)
        self.obs_history[:] = 0.0
        self.last_action[:] = 0.0
        self.target_joints = self.policy.default_joint_pos.copy()
        self.phys_step = 0


# ---------------------------------------------------------------------------
# TODO 1：HSV 红球检测
# ---------------------------------------------------------------------------


def detect_red_ball(rgb: np.ndarray) -> dict | None:
    """检测图像中的红球，返回 bbox dict 或 None。

    Parameters
    ----------
    rgb : (H, W, 3) uint8 RGB 图像

    Returns
    -------
    dict with keys cx, cy, w, h, area  或  None（未检测到）

    提示
    ----
    1. RGB → BGR → HSV
    2. 红色在 HSV 横跨 0 度，需要两段 inRange 取并集：
       - 段 1：H ∈ [0, 10], S ∈ [120, 255], V ∈ [80, 255]
       - 段 2：H ∈ [170, 180], S ∈ [120, 255], V ∈ [80, 255]
    3. findContours → 取面积最大的轮廓
    4. 面积 < _MIN_AREA 视为噪点，返回 None
    5. boundingRect → 返回 {cx, cy, w, h, area}
    """
    # ===== TODO 1 START =====
    raise NotImplementedError("TODO 1: 实现 HSV 红球检测")
    # ===== TODO 1 END =====


# ---------------------------------------------------------------------------
# TODO 2：P 控制器 + FSM
# ---------------------------------------------------------------------------


class TrackerFSM:
    """三态追踪状态机：SEARCHING → TRACKING → STOPPED。

    提示
    ----
    - visual_servo(box, W, H) → (vx, wz)：
      e_yaw = (cx - W/2) / (W/2)
      wz = -KP_YAW * e_yaw
      e_size = max(0, 1 - box_h / H_REF)
      vx = KP_VX * e_size

    - FSM 三态切换：
      - 检测到球 → TRACKING，用 visual_servo 计算命令
      - 丢失 > LOST_TIMEOUT → SEARCHING，原地慢转 (0, SEARCH_WZ)
      - 丢失 > LOST_TIMEOUT * 3 → STOPPED，(0, 0)
    """

    def __init__(self) -> None:
        self.state = "SEARCHING"
        self._last_seen: float = 0.0
        self._last_box: dict | None = None

    def reset(self) -> None:
        self.state = "SEARCHING"
        self._last_seen = 0.0
        self._last_box = None

    def step(
        self,
        t: float,
        boxes: list[dict],
        image_w: int,
        image_h: int,
    ) -> tuple[str, float, float]:
        """根据当前检测结果更新状态，返回 (state, vx, wz)。"""
        # ===== TODO 2 START =====
        raise NotImplementedError("TODO 2: 实现 P 控制器 + FSM 三态切换")
        # ===== TODO 2 END =====


# ---------------------------------------------------------------------------
# TODO 3：追球主循环
# ---------------------------------------------------------------------------


def run_chase(
    robot: ChaseRobotState,
    head_renderer: mujoco.Renderer,
    head_w: int = 320,
    head_h: int = 240,
    duration: float = CHASE_DURATION,
    *,
    frame_callback=None,
) -> dict:
    """追球闭环：camera → detect → fsm.step → (vx, wz) → policy → mj_step。

    提示
    ----
    主循环结构（每个物理步 500 Hz）：
      1. 更新球位置（ball_mover.update）
      2. 如果到了视觉采样时间（10 Hz）：
         - head_renderer.update_scene + render → rgb
         - detect_red_ball(rgb) → box
         - fsm.step(t, [box] if box else [], W, H) → (state, vx, wz)
      3. 每 CONTROL_RATIO 步：robot._policy_tick(cmd)
      4. robot.data.ctrl[:] = robot.target_joints
      5. mujoco.mj_step(robot.model, robot.data)
      6. robot.phys_step += 1
      7. 如果 frame_callback：调用它
      8. 如果 robot.is_fallen()：提前返回

    Returns
    -------
    dict with keys: timeline (list of dicts), ok (bool)
    """
    # ===== TODO 3 START =====
    raise NotImplementedError("TODO 3: 实现追球主循环")
    # ===== TODO 3 END =====


# ---------------------------------------------------------------------------
# 球运动控制（已实现，不需要修改）
# ---------------------------------------------------------------------------


class BallMover:
    """控制 mocap 红球的随机运动（线性插值，不瞬移）。"""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng(42)
        self._target = np.array([1.5, 0.3, 0.05])
        self._origin = self._target.copy()
        self._next_move_time = BALL_MOVE_INTERVAL
        self._move_start_time = 0.0
        self._move_duration = 1.0

    def update(self, t: float, data: mujoco.MjData) -> None:
        if t >= self._next_move_time:
            self._origin = data.mocap_pos[0].copy()
            self._target = np.array([
                self._rng.uniform(1.0, 3.0),
                self._rng.uniform(-1.0, 1.0),
                0.05,
            ])
            self._move_start_time = t
            self._move_duration = 1.0
            self._next_move_time = t + BALL_MOVE_INTERVAL

        alpha = min(1.0, (t - self._move_start_time) / self._move_duration)
        data.mocap_pos[0] = self._origin + alpha * (self._target - self._origin)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    """快速环境检查。"""
    robot = ChaseRobotState(policy_name="test")
    robot.reset()
    print(f"Lab 8 环境检查: policy={robot.policy.path.name}")

    result = robot.step_cmd(0.5, 0.0, 2.0)
    print(f"step_cmd(0.5, 0, 2.0): {result}")
    print("Lab 8 环境检查通过。物理系统正常。")
    print("请补全 TODO 1–3 后运行 eval_chase.py 生成交付物。")


if __name__ == "__main__":
    main()
