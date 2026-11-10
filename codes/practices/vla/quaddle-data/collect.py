"""Quaddle VLA 数据采集：脚本化专家 + MuJoCo 渲染 + LeRobotDataset。

任务：桌面上散布 3 个彩色积木（red/green/blue），指令指定目标颜色，
机器人须走到该积木旁停下。专家用查表 CPG 控制器（`8.rl_quaddle/
cpg_controller.py`，W2 交付物）做方位闭环：

    |bearing| < 25° 且 dist > 0.12m → forward
    bearing > +25°                  → turn_left
    bearing < -25°                  → turn_right
    dist < 0.12m                    → stop（成功）

数据（fps=10，物理 1kHz / 控制 50Hz / 每 5 控制步记 1 帧）：
    observation.image  320×320 overhead RGB（固定俯视相机）
    observation.state  [4 舵机角, vx, wz, bearing, dist]（8 维）
    action             [vx_dir, wz_dir] ∈ {-1,0,+1}²（档位指令）
    task               "walk to the {color} block"

失败 episode（翻倒/超时 45s）整段丢弃，不入库。

运行（必须用 vla/act 的 lerobot 环境）：
    cd codes/practices/vla/act
    MUJOCO_GL=egl uv run python ../quaddle-data/collect.py \
        --episodes 8 --repo-id local/quaddle_nav_smoke
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CS123 = HERE.parent.parent / "quadruped" / "cs123"
QUADDLE_DIR = CS123 / "7.quaddle-mjcf"
QUADDLE_XML = QUADDLE_DIR / "quaddle.xml"
sys.path.insert(0, str(CS123 / "8.rl_quaddle"))

import mujoco  # noqa: E402

COLORS = ("red", "green", "blue")
COLOR_RGBA = {"red": (0.9, 0.2, 0.2, 1), "green": (0.2, 0.8, 0.3, 1), "blue": (0.25, 0.4, 0.9, 1)}
OBJ_RNG = (0.28, 0.70)      # 目标极径范围（m）：更远 → 每条路径更多档位切换
MIN_OBJ_GAP = 0.22          # 积木间最小间距
SUCCESS_DIST = 0.12         # 到达判定（含 stop 滑行余量）
# 专家死区每回合随机化（改进路径 1）：固定死区使 turn→forward 切换边界
# 在数据中过稀（v2 训练数据实际仍是固定 ±25°，ep0/1 转向锁死归因之一）
BEARING_DEADZONE_MIN = math.radians(12.0)
BEARING_DEADZONE_MAX = math.radians(40.0)
BEARING_DEADZONE = math.radians(25.0)  # 保留兼容旧引用
TIMEOUT_S = 45.0
CTRL_HZ = 50
DATA_HZ = 10                # 数据帧率
CAM_H = 1.4                 # overhead 相机高度（覆盖 ~1.15m 视野）
IMG = 320
SETTLE_S = 1.2


def build_scene(out_xml: Path) -> None:
    """在 quaddle.xml 基础上注入 3 个积木位与 overhead 相机，写 scene.xml。"""
    tree = ET.parse(QUADDLE_XML)
    root = tree.getroot()
    # mesh 引用改为绝对路径（scene.xml 存放在别的目录）
    for mesh in root.find("asset").findall("mesh"):
        mesh.set("file", str((QUADDLE_DIR / mesh.get("file")).resolve()))
    world = root.find("worldbody")
    for i, color in enumerate(COLORS):
        ET.SubElement(
            world, "geom",
            name=f"block_{color}", type="box", size="0.025 0.025 0.025",
            rgba=" ".join(f"{v:g}" for v in COLOR_RGBA[color]),
            pos="0 0 0.025", contype="1", conaffinity="1",
        )
    ET.SubElement(world, "camera", name="overhead", pos=f"0 0 {CAM_H}",
                  xyaxes="1 0 0 0 1 0")
    # SmolVLA 基座特征签名要求 camera1/2/3 三路视觉输入
    ET.SubElement(world, "camera", name="front",
                  pos="1.3 0 0.28", xyaxes="0 1 0 -0.35 0 0.94")
    ET.SubElement(world, "camera", name="oblique",
                  pos="-0.75 0.75 1.1", xyaxes="0.7 0.7 0 -0.45 -0.45 0.77")
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="unicode", xml_declaration=False)


def random_layout(rng) -> list[tuple[float, float]]:
    """3 个积木的极坐标布点（间隔约束）。"""
    for _ in range(200):
        r = rng.uniform(*OBJ_RNG)
        th = rng.uniform(-math.pi, math.pi)
        pts = [(r * math.cos(th), r * math.sin(th))]
        ok = True
        for _ in range(2):
            r2 = rng.uniform(*OBJ_RNG)
            th2 = rng.uniform(-math.pi, math.pi)
            p2 = (r2 * math.cos(th2), r2 * math.sin(th2))
            if all(math.dist(p2, q) >= MIN_OBJ_GAP for q in pts):
                pts.append(p2)
            else:
                ok = False
                break
        if ok and len(pts) == 3:
            return pts
    raise RuntimeError("布点失败")


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--repo-id", default="local/quaddle_nav_smoke")
    ap.add_argument("--root", default=str(HERE / "outputs" / "dataset"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-tries", type=int, default=3, help="每个目标 episode 的最大尝试数")
    args = ap.parse_args()

    # lerobot 导入（在 vla/act 环境中运行）
    from lerobot.datasets import LeRobotDataset

    out_dir = Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = out_dir / "scene.xml"
    build_scene(scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    # CPG 控制器（复用 W2 交付物；仅用其 SKILLS/step，reset 由本脚本管理）
    from cpg_controller import CpgController
    # 固定 overhead 渲染器
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    renderer = mujoco.Renderer(model, height=IMG, width=IMG)

    features = {
        # names 不能为 None：dataset_to_policy_features 需要按 names 识别 HWC→CHW
        "observation.images.camera1": {"dtype": "video", "shape": (IMG, IMG, 3),
                              "names": ["height", "width", "channel"]},
        "observation.images.camera2": {"dtype": "video", "shape": (IMG, IMG, 3),
                              "names": ["height", "width", "channel"]},
        "observation.images.camera3": {"dtype": "video", "shape": (IMG, IMG, 3),
                              "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["action"]},
    }
    ds_root = out_dir / "lerobot"
    if ds_root.exists():
        import shutil
        shutil.rmtree(ds_root)  # 重跑覆盖（采集脚本语义：全新数据集）
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=DATA_HZ,
        features=features,
        root=str(out_dir / "lerobot"),
        robot_type="quaddle",
        use_videos=True,
    )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {
        c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
        for c in COLORS
    }
    n_ctrl_per_frame = int(CTRL_HZ // DATA_HZ)  # 5 控制步 = 1 数据帧（10Hz）

    rng = np.random.default_rng(args.seed)
    n_saved, n_fail = 0, 0
    ep = 0
    while n_saved < args.episodes and ep < args.episodes * 6:  # 上限防死循环
        ep += 1
        color = COLORS[ep % 3]
        deadzone = float(rng.uniform(BEARING_DEADZONE_MIN, BEARING_DEADZONE_MAX))
        layout = random_layout(rng)
        target_xy = layout[COLORS.index(color)]
        yaw0 = float(rng.uniform(-math.pi, math.pi))

        # --- reset：home keyframe + 随机 yaw + 布点 + 沉降 ---
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[3:7] = yaw_quat(yaw0)
        for i, c in enumerate(COLORS):
            gid = geom_ids[c]
            model.geom_pos[gid][:2] = layout[i]
        mujoco.mj_forward(model, data)
        data.ctrl[:] = 0.0
        for _ in range(int(SETTLE_S / model.opt.timestep)):
            mujoco.mj_step(model, data)

        # 控制器状态复位（绕过 __init__ 的沉降——本脚本自行管理 reset）
        import types
        ctrl = CpgController.__new__(CpgController)
        ctrl.env = types.SimpleNamespace(n_substeps=20)
        ctrl.model, ctrl.data = model, data
        ctrl.dt = 0.02
        ctrl._phases = np.zeros(4)   # 每腿独立相位钟（cpg_controller v2）
        ctrl._amp_scale, ctrl._dir = 0.0, 1.0
        ctrl.skill = "stop"

        tgt = np.array([target_xy[0], target_xy[1], 0.0])
        frames: list[dict] = []
        success, t_sim = False, 0.0
        while t_sim < TIMEOUT_S:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            d_world = tgt - pos
            dist = float(np.linalg.norm(d_world[:2]))
            d_body = r.T @ d_world
            bearing = math.atan2(d_body[0], -d_body[1])  # 左正；前向 = -body_y

            if dist < SUCCESS_DIST:
                skill, act = "stop", (0.0, 0.0)
                success = True
            elif bearing > deadzone:
                # 转向动作连续强度编码：bearing 越大越强（0.35~1.0），
                # 匹配 flow matching 保守输出特性（W5 首测 ep0/1 弧线死循环归因）
                strength = min(1.0, 0.35 + abs(bearing) / math.radians(90))
                skill, act = "turn_left", (0.0, strength)
            elif bearing < -deadzone:
                strength = min(1.0, 0.35 + abs(bearing) / math.radians(90))
                skill, act = "turn_right", (0.0, -strength)
            else:
                skill, act = "forward", (1.0, 0.0)
            if skill.startswith("turn"):
                ctrl.set_turn_skill(skill.split("_")[1], abs(act[1]))
            else:
                ctrl.set_skill(skill)

            for _ in range(n_ctrl_per_frame):
                ctrl.step()
                t_sim += ctrl.dt
                if data.body(body_id).xpos[2] < 0.030:
                    break
            if data.body(body_id).xpos[2] < 0.030:
                break

            # 记录帧（10Hz）：三路相机（camera1=overhead 主视角）
            frames_img = {}
            for cam_name, feat in (("overhead", "camera1"), ("front", "camera2"), ("oblique", "camera3")):
                renderer.update_scene(data, camera=cam_name)
                frames_img[feat] = renderer.render().copy()
            img = frames_img["camera1"]
            joint_q = np.array([data.qpos[7 + i * 6] for i in range(4)])
            v_body = r.T @ data.qvel[:3]
            state = np.concatenate([
                joint_q, [float(-v_body[1]), float((r.T @ data.qvel[3:6])[2])],
                [bearing, dist],
            ]).astype(np.float32)
            frames.append({
                "observation.images.camera1": frames_img["camera1"],
                "observation.images.camera2": frames_img["camera2"],
                "observation.images.camera3": frames_img["camera3"],
                "observation.state": state,
                "action": np.array(act, dtype=np.float32),
                "task": f"walk to the {color} block",
            })
            if success:
                break

        if success and frames:
            for fr in frames:
                dataset.add_frame(fr)
            dataset.save_episode()
            n_saved += 1
            print(f"ep{ep:03d} ✓ {color:5s} 目标{np.linalg.norm(target_xy):.2f}m "
                  f"→ {t_sim:5.1f}s，{len(frames)} 帧（成功 {n_saved}/{args.episodes}，丢弃 {n_fail}）", flush=True)
        else:
            n_fail += 1
            reason = "timeout" if t_sim >= TIMEOUT_S else "fallen"
            print(f"ep{ep:03d} ✗ {color:5s} 丢弃（{reason}，{t_sim:.1f}s）", flush=True)

    dataset.finalize()
    print(f"\n完成：保存 {n_saved} / 丢弃 {n_fail}，数据集 {out_dir / 'lerobot'}")


if __name__ == "__main__":
    main()
