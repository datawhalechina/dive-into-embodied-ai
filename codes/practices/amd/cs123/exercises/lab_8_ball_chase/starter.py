"""Lab 8 filled starter：追球 Demo（Ball-Chasing Demo）。

教师版完整实现：在 pupper_chase.xml 场景中，用 HSV 检测红球 → P 控制器 →
TrackerFSM 三态切换 → RL 步态闭环，录制追球 GIF + FSM 时间线 + 图表。

物理后端严格复用 Lab 7 的 RobotState（test_policy.json + 50 Hz / 500 Hz），
只替换 XML 路径为 pupper_chase.xml（加了 head_cam + mocap 红球）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from perception.ball_detector import detect_red_ball  # noqa: E402
from perception.tracker_fsm import TrackerFSM, FSMState  # noqa: E402

GIF_FPS = 10
GIF_WIDTH = 480
GIF_HEIGHT = 360
VISION_HZ = 10
CHASE_DURATION = 20.0
BALL_MOVE_INTERVAL = 3.0

# ---------------------------------------------------------------------------
# ChaseRobotState — 只替换 XML 路径，其余完全继承 Lab 7
# ---------------------------------------------------------------------------


class ChaseRobotState(RobotState):
    """Lab 7 RobotState 的子类，唯一区别是加载 pupper_chase.xml。

    _settle / _policy_tick / step_cmd / is_fallen / get_pose 全部继承，
    不做任何修改——这是 prompt 里最重要的约束。
    """

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
# 渲染工具
# ---------------------------------------------------------------------------


def _setup_renderer(robot: ChaseRobotState) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    robot.model.vis.global_.offwidth = max(robot.model.vis.global_.offwidth, GIF_WIDTH)
    robot.model.vis.global_.offheight = max(robot.model.vis.global_.offheight, GIF_HEIGHT)
    renderer = mujoco.Renderer(robot.model, height=GIF_HEIGHT, width=GIF_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(robot.model, camera)
    camera.distance = 3.0
    camera.elevation = -25.0
    camera.azimuth = 135.0
    return renderer, camera


def _setup_head_renderer(robot: ChaseRobotState, w: int = 320, h: int = 240) -> mujoco.Renderer:
    robot.model.vis.global_.offwidth = max(robot.model.vis.global_.offwidth, w)
    robot.model.vis.global_.offheight = max(robot.model.vis.global_.offheight, h)
    return mujoco.Renderer(robot.model, height=h, width=w)


def _caption_chase_frame(
    frame: np.ndarray,
    fsm_state: FSMState,
    vx: float,
    wz: float,
    t: float,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(20)
    small = gif_utils.load_font(14)

    state_text_colors = {
        FSMState.SEARCHING: (200, 160, 0, 255),
        FSMState.TRACKING: (0, 160, 60, 255),
        FSMState.STOPPED: (200, 60, 60, 255),
    }
    text_color = state_text_colors.get(fsm_state, (200, 60, 60, 255))
    draw.rectangle((0, 0, image.width, 52), fill=(255, 255, 255, 220))
    draw.text((10, 4), f"[{fsm_state.value}]", fill=text_color, font=font)
    draw.text((10, 28), f"vx={vx:.2f}  wz={wz:.2f}", fill=(18, 30, 44, 255), font=small)

    draw.rectangle((0, image.height - 32, image.width, image.height), fill=(0, 0, 0, 160))
    draw.text(
        (10, image.height - 28),
        f"Ball Chase  t={t:.1f}s",
        fill=(230, 230, 230, 240),
        font=small,
    )
    return np.asarray(image)


# ---------------------------------------------------------------------------
# 球运动控制
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
# 追球主循环
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

    Returns
    -------
    dict with keys: timeline (list of dicts), ok (bool)
    """
    fsm = TrackerFSM()
    ball_mover = BallMover()
    head_cam_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")

    vision_interval = 1.0 / VISION_HZ
    next_vision_time = 0.0
    current_vx, current_wz = 0.0, 0.0
    current_state = FSMState.SEARCHING

    timeline: list[dict] = []
    n_phys = int(duration / PHYS_DT)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for s in range(n_phys):
        t = robot.sim_time

        ball_mover.update(t, robot.data)

        if t >= next_vision_time:
            head_renderer.update_scene(robot.data, camera=head_cam_id)
            rgb = head_renderer.render()
            box = detect_red_ball(rgb)
            boxes = [box] if box is not None else []
            current_state, current_vx, current_wz = fsm.step(t, boxes, head_w, head_h)
            next_vision_time = t + vision_interval

            ball_pos = robot.data.mocap_pos[0].copy()
            robot_pos = robot.data.qpos[0:3].copy()
            dist = float(np.linalg.norm(ball_pos[:2] - robot_pos[:2]))
            timeline.append({
                "t": t,
                "state": current_state.value,
                "vx": current_vx,
                "wz": current_wz,
                "e_yaw": (box["cx"] - head_w / 2) / (head_w / 2) if box else None,
                "box_h": box["h"] if box else None,
                "dist": dist,
            })

        cmd[0] = current_vx
        cmd[2] = current_wz

        if s % CONTROL_RATIO == 0:
            robot._policy_tick(cmd)
        robot.data.ctrl[:] = robot.target_joints
        mujoco.mj_step(robot.model, robot.data)
        robot.phys_step += 1

        if frame_callback is not None:
            frame_callback(robot, current_state, current_vx, current_wz)

        if robot.is_fallen():
            return {"ok": False, "error": "robot fell", "timeline": timeline}

    return {"ok": True, "timeline": timeline}


# ---------------------------------------------------------------------------
# GIF 录制
# ---------------------------------------------------------------------------


def record_chase_gif(
    robot: ChaseRobotState,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    head_renderer: mujoco.Renderer,
    head_w: int = 320,
    head_h: int = 240,
    duration: float = CHASE_DURATION,
) -> tuple[list[np.ndarray], dict]:
    """录制追球 GIF，返回 (frames, result)。"""
    robot.reset()

    frames: list[np.ndarray] = []
    t_start = robot.sim_time
    next_frame_time = [0.0]

    def frame_callback(
        cb_robot: ChaseRobotState,
        state: FSMState,
        vx: float,
        wz: float,
    ):
        t = cb_robot.sim_time - t_start
        if t >= next_frame_time[0]:
            robot_pos = cb_robot.data.xpos[cb_robot.base_id]
            ball_pos = cb_robot.data.mocap_pos[0]
            midpoint = (robot_pos + ball_pos) / 2.0
            midpoint[2] = 0.15
            camera.lookat[:] = midpoint
            renderer.update_scene(cb_robot.data, camera=camera)
            raw = renderer.render()
            frames.append(_caption_chase_frame(raw, state, vx, wz, t))
            next_frame_time[0] += 1.0 / GIF_FPS

    result = run_chase(
        robot, head_renderer, head_w, head_h, duration,
        frame_callback=frame_callback,
    )
    return frames, result


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------


def plot_tracking_error(timeline: list[dict], out_path: Path) -> None:
    """e_yaw vs 时间，叠加 FSM 状态色条。"""
    ts = [r["t"] for r in timeline]
    e_yaws = [r["e_yaw"] for r in timeline]
    states = [r["state"] for r in timeline]

    fig, ax = plt.subplots(figsize=(10, 3))

    state_colors = {"SEARCHING": "#FFC800", "TRACKING": "#00C850", "STOPPED": "#C83C3C"}
    for i in range(len(ts) - 1):
        ax.axvspan(ts[i], ts[i + 1], alpha=0.15, color=state_colors.get(states[i], "#888"))

    valid_ts = [t for t, e in zip(ts, e_yaws) if e is not None]
    valid_es = [e for e in e_yaws if e is not None]
    ax.plot(valid_ts, valid_es, "b-", linewidth=1.0, label="e_yaw")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("e_yaw (normalized)")
    ax.set_title("Tracking Error vs Time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_detection_vs_distance(timeline: list[dict], out_path: Path) -> None:
    """球到机器人距离 vs bbox 高度。"""
    dists = [r["dist"] for r in timeline if r["box_h"] is not None]
    box_hs = [r["box_h"] for r in timeline if r["box_h"] is not None]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(dists, box_hs, s=8, alpha=0.6, label="detections")
    from perception.tracker_fsm import H_REF
    ax.axhline(H_REF, color="red", linewidth=1.0, linestyle="--", label=f"H_REF={H_REF}")
    ax.set_xlabel("distance to ball (m)")
    ax.set_ylabel("bbox height (px)")
    ax.set_title("Detection vs Distance")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    """快速环境检查（不需要 API key）。"""
    robot = ChaseRobotState(policy_name="test")
    robot.reset()
    print(f"Lab 8 环境检查: policy={robot.policy.path.name}")

    result = robot.step_cmd(0.5, 0.0, 2.0)
    print(f"step_cmd(0.5, 0, 2.0): {result}")

    head_renderer = _setup_head_renderer(robot)
    head_cam_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    head_renderer.update_scene(robot.data, camera=head_cam_id)
    rgb = head_renderer.render()
    print(f"head_cam render: shape={rgb.shape}, dtype={rgb.dtype}")

    box = detect_red_ball(rgb)
    print(f"detect_red_ball: {box}")
    head_renderer.close()

    print("Lab 8 环境检查通过。")


if __name__ == "__main__":
    main()
