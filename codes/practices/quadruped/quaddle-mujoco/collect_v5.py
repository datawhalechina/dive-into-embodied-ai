"""Quaddle VLA v5 数据采集：《Quaddle 入门》环境（quaddle-mujoco）+ steer 专家。

环境统一决策（2026-11-11）：VLA 管线从 cs123/7.quaddle-mjcf 迁移到本环境。
迁移依据（spin_feasibility_v5.py + expert_smoke_v5.py 实测）：
  - 本环境原生转向 = duty 不对称（steer_limit），实测 wz ±0.6~1.2 rad/s、
    转弯半径 ~2.6cm（准原地转）、连续 10s 不翻——无需 spin 旗标，
    动作空间回退 2 维 [vx_dir, wz_dir]
  - 专家闭环两 seed 全胜：9/9 + 12/12 到达、零翻倒（含 rear 场景过半）

任务与数据规格（与 v4 一致，仅动作 3→2 维）：
    "walk to the {color} block"，3 积木散布，12cm 内刹停
    observation.images.camera1/2/3  320² RGB（overhead/front/oblique）
    observation.state  [4 舵机角, vx, wz, bearing, dist]
    action             [vx_dir, wz_dir]：forward=(1,0) steer_left=(0,1)
                       steer_right=(0,-1) stop=(0,0)
    初始朝向 50% 过采样目标正后方；死区 ±12~40° 每回合随机；切档驻留 0.4s

运行（leroenv）：
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib \
      HF_HUB_OFFLINE=1 python collect_v5.py --episodes 400 \
      --repo-id local/quaddle_nav_400v5 --seed 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE / "models" / "scene.xml"
CONFIG = json.loads((HERE / "gait.json").read_text())

DT_PHYS, DT_CTRL = 0.001, 0.02
SUB = 20
SETTLE_S = float(CONFIG.get("settle_seconds", 2.0))
RAMP_S = float(CONFIG.get("ramp_seconds", 1.5))
CTRL_MAX = 1.2
SIGNS = np.array(CONFIG["actuator_signs"])
BIAS = np.array(CONFIG["bias_rad"])
AMPS = np.array(CONFIG["amplitude_rad"])
PHASE0 = np.array(CONFIG["phase_rad"])
FREQ = float(CONFIG["frequency_hz"])
DUTY = float(CONFIG["sweep_fraction"])
STEER_LIMIT = float(CONFIG["steering_limit"])
SIDE = {"L": (0, 3), "R": (1, 2)}

COLORS = ("red", "green", "blue")
COLOR_RGBA = {"red": (0.9, 0.2, 0.2, 1), "green": (0.2, 0.8, 0.3, 1), "blue": (0.25, 0.4, 0.9, 1)}
OBJ_RNG = (0.28, 0.70)
MIN_GAP = 0.22
SUCCESS_DIST = 0.12
BEARING_DEADZONE_MIN = math.radians(12.0)
BEARING_DEADZONE_MAX = math.radians(40.0)
REAR_OVERSAMPLE = 0.5
DWELL_S = 0.4
TIMEOUT_S = 45.0
CTRL_HZ = 50
DATA_HZ = 10
IMG = 320


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


def build_scene(out_xml: Path) -> None:
    """scene.xml 副本（写到 dataset 目录）+ meshdir 转绝对路径 + 注入 3 积木 + 三相机。"""
    tree = ET.parse(XML)
    for compiler in tree.getroot().iter("compiler"):
        md = compiler.get("meshdir")
        if md and not os.path.isabs(md):
            compiler.set("meshdir", str((XML.parent / md).resolve()))
    world = tree.getroot().find("worldbody")
    for c in COLORS:
        ET.SubElement(world, "geom", name=f"block_{c}", type="box",
                      size="0.025 0.025 0.025",
                      rgba=" ".join(f"{v:g}" for v in COLOR_RGBA[c]),
                      pos="0 0 0.025", contype="1", conaffinity="1")
    ET.SubElement(world, "camera", name="overhead", pos="0 0 1.4", xyaxes="1 0 0 0 1 0")
    ET.SubElement(world, "camera", name="front", pos="1.3 0 0.28", xyaxes="0 1 0 -0.35 0 0.94")
    ET.SubElement(world, "camera", name="oblique", pos="-0.75 0.75 1.1", xyaxes="0.7 0.7 0 -0.45 -0.45 0.77")
    tree.write(out_xml, encoding="unicode")


def random_layout(rng):
    for _ in range(200):
        pts = []
        while len(pts) < 3:
            r = rng.uniform(*OBJ_RNG)
            th = rng.uniform(-math.pi, math.pi)
            p = (r * math.cos(th), r * math.sin(th))
            if all(math.dist(p, q) >= MIN_GAP for q in pts):
                pts.append(p)
        return pts
    raise RuntimeError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--repo-id", default="local/quaddle_nav_400v5")
    ap.add_argument("--root", default=str(HERE / "outputs" / "dataset"))
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--recovery", type=float, default=0.0,
                    help="恢复数据模式：专家接管前执行 1~RECOVERY 秒随机档位扰动"
                         "（不录制），覆盖自旋转/偏移的 OOD 状态 + 专家恢复行为")
    ap.add_argument("--action-1d", action="store_true",
                    help="#3 单通道档位动作：mode∈{-1,0,+1}={steer_R,forward/stop,steer_L}，"
                         "消除 vx/wz 联合外推空间（默认 2 维 [vx_dir,wz_dir]）")
    ap.add_argument("--warmup-min", type=float, default=1.0,
                    help="恢复扰动最短时长（#4 扩量设 15）")
    ap.add_argument("--warmup-safety", action="store_true",
                    help="#4 防翻安全偏置：warmup 中监测 body 倾斜，超阈值强制前进档稳定，"
                         "允许更长自旋态而不翻倒")
    ap.add_argument("--gait-freq", type=float, default=0.0,
                    help="覆盖步态频率 Hz（0=使用 gait.json 默认值）")
    ap.add_argument("--gait-amp-scale", type=float, default=1.0,
                    help="步态振幅缩放因子（1.0=原始）")
    ap.add_argument("--servo-kp", type=float, default=0.0,
                    help="覆盖舵机 kp（0=scene.xml 默认 0.15）；快步态需更高 kp 才能追上目标")
    ap.add_argument("--servo-kv", type=float, default=0.0,
                    help="覆盖舵机 kv 阻尼（0=默认 0.006）")
    ap.add_argument("--dwell-s", type=float, default=0.0,
                    help="覆盖切档驻留秒（0=默认 0.4；快步态需更短防过冲振荡）")
    ap.add_argument("--deadzone-min-deg", type=float, default=0.0,
                    help="覆盖 bearing 死区下限度（0=默认 12）")
    ap.add_argument("--deadzone-max-deg", type=float, default=0.0,
                    help="覆盖 bearing 死区上限度（0=默认 40）")
    args = ap.parse_args()

    global FREQ, AMPS, DWELL_S, BEARING_DEADZONE_MIN, BEARING_DEADZONE_MAX
    if args.gait_freq > 0:
        FREQ = args.gait_freq
    if args.gait_amp_scale != 1.0:
        AMPS = AMPS * args.gait_amp_scale
    if args.dwell_s > 0:
        DWELL_S = args.dwell_s
    if args.deadzone_min_deg > 0:
        BEARING_DEADZONE_MIN = math.radians(args.deadzone_min_deg)
    if args.deadzone_max_deg > 0:
        BEARING_DEADZONE_MAX = math.radians(args.deadzone_max_deg)

    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.datasets import LeRobotDataset

    out_dir = Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = out_dir / "scene.xml"
    build_scene(scene_xml)

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = DT_PHYS
    if args.servo_kp > 0:
        model.actuator_gainprm[:, 0] = args.servo_kp
        model.actuator_biasprm[:, 1] = -args.servo_kp
        model.actuator_forcerange[:] = [-max(0.245, args.servo_kp * 0.5), max(0.245, args.servo_kp * 0.5)]
    if args.servo_kv > 0:
        model.actuator_biasprm[:, 2] = -args.servo_kv
    data = mujoco.MjData(model)

    # 舵机 qpos 地址：按执行器顺序（lf, rf, rb, lb）解析 trnid
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in COLORS}
    renderer = mujoco.Renderer(model, height=IMG, width=IMG)

    features = {
        f"observation.images.camera{i}": {"dtype": "video", "shape": (IMG, IMG, 3),
                                          "names": ["height", "width", "channel"]}
        for i in (1, 2, 3)
    }
    features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    action_dim = 1 if args.action_1d else 2
    features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": ["action"]}
    ds_root = out_dir / "lerobot"
    if ds_root.exists():
        import shutil
        shutil.rmtree(ds_root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=DATA_HZ, features=features,
        root=str(ds_root), robot_type="quaddle", use_videos=True,
    )

    rng = np.random.default_rng(args.seed)
    n_saved, n_fail, ep = 0, 0, 0
    if args.action_1d:
        # mode: steer_R=-1, forward=0, steer_L=+1, stop=0（并入前进，eval 以 dist<12cm 判停）
        ACTIONS = {"forward": (0.0,), "steer_L": (1.0,),
                   "steer_R": (-1.0,), "stop": (0.0,)}
    else:
        ACTIONS = {"forward": (1.0, 0.0), "steer_L": (0.0, 1.0),
                   "steer_R": (0.0, -1.0), "stop": (0.0, 0.0)}

    while n_saved < args.episodes and ep < args.episodes * 6:
        ep += 1
        color = COLORS[ep % 3]
        deadzone = float(rng.uniform(BEARING_DEADZONE_MIN, BEARING_DEADZONE_MAX))
        layout = random_layout(rng)
        tgt_xy = layout[COLORS.index(color)]
        th_t = math.atan2(tgt_xy[1], tgt_xy[0])
        if rng.random() < REAR_OVERSAMPLE:
            yaw0 = th_t - math.pi / 2 + rng.uniform(-math.radians(30), math.radians(30))
        else:
            yaw0 = float(rng.uniform(-math.pi, math.pi))

        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
        for i, c in enumerate(COLORS):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(int(SETTLE_S / DT_PHYS)):
            mujoco.mj_step(model, data)

        # 恢复数据模式：专家接管前随机扰动（不录制）——把机器人推进自旋转/
        # 偏移等 OOD 状态，随后的专家恢复段教会策略"从乱态回到目标"
        if args.recovery > 0:
            warm = float(rng.uniform(args.warmup_min, args.recovery))
            # #4 偏向转向档（70%）诱发原地自旋 OOD 态：准原地转（半径~2.6cm）保持远离
            # 目标，专家接管后有真实恢复轨迹；纯随机前进会晃到目标旁导致 1 帧空 episode
            def pick_gear():
                return int(rng.integers(1, 3)) if rng.random() < 0.7 else 0
            wt, wgear, wgswitch = 0.0, pick_gear(), float(rng.uniform(0.8, 2.0))
            wamp, wphases = 0.0, 0.0
            while wt < warm:
                if wt > wgswitch:
                    wgear = pick_gear()
                    wgswitch = wt + float(rng.uniform(0.8, 2.0))
                # #4 防翻安全偏置：body z 轴竖直分量 r[2,2]=cos(tilt)，倾斜超 ~23°
                # 强制前进档稳定，待回正再恢复随机档——允许更长自旋态而不翻倒
                if args.warmup_safety:
                    rz = float(data.body(body_id).xmat.reshape(3, 3)[2, 2])
                    if rz < 0.92:
                        wgear = 0
                        wgswitch = wt + 0.6
                wamp = min(1.0, wamp + DT_CTRL / RAMP_S)
                wphases += 2 * math.pi * FREQ * DT_CTRL
                sduty = np.full(4, DUTY)
                if wgear == 1:
                    sduty[list(SIDE["L"])] *= (1 - STEER_LIMIT)
                    sduty[list(SIDE["R"])] *= (1 + STEER_LIMIT)
                elif wgear == 2:
                    sduty[list(SIDE["L"])] *= (1 + STEER_LIMIT)
                    sduty[list(SIDE["R"])] *= (1 - STEER_LIMIT)
                wamps = AMPS * wamp
                data.ctrl[:] = np.clip(SIGNS * (BIAS + wamps * wave_at(wphases + PHASE0, sduty)),
                                       -CTRL_MAX, CTRL_MAX)
                fell_warm = False
                for _ in range(SUB):
                    mujoco.mj_step(model, data)
                wt += DT_CTRL
                if data.body(body_id).xpos[2] < 0.030:
                    fell_warm = True
                    break
            if fell_warm:
                n_fail += 1
                print(f"ep{ep:03d} ✗ {color:5s} 丢弃（warmup 翻倒，{warm:.1f}s）", flush=True)
                continue
            # 扰动后重沉降半秒，防专家从弹跳态起步
            data.ctrl[:] = 0.0
            for _ in range(int(0.5 / DT_PHYS)):
                mujoco.mj_step(model, data)
            # #4 距目标过近则丢弃：保证恢复段有真实接近轨迹（避免 1 帧空 episode）
            pos_w = data.body(body_id).xpos
            dist_w = float(np.linalg.norm(tgt_xy - pos_w[:2]))
            if dist_w < 0.20:
                n_fail += 1
                print(f"ep{ep:03d} ✗ {color:5s} 丢弃（warmup 后距目标 {dist_w:.2f}m 过近）", flush=True)
                continue

        amp_scale, phases_t = 0.0, 0.0
        skill = skill0 = "forward"
        last_switch_t, steer_now, duty = 0.0, 0.0, np.full(4, DUTY)
        frames = []
        success, t_sim = False, 0.0

        while t_sim < TIMEOUT_S:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            d_world = np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
            dist = float(np.linalg.norm(d_world[:2]))
            d_body = r.T @ d_world
            bearing = math.atan2(d_body[0], -d_body[1])  # 左正（与 cs123 同约定）

            if dist < SUCCESS_DIST:
                skill = "stop"
                success = True
            elif abs(bearing) > deadzone:
                # 二值档位（v4 教训：连续强度=弱执行易超时；本环境 CW 方向
                # 上限 0.61 rad/s——±0.33 以上不对称会翻倒，scan5b 实测）
                skill = "steer_L" if bearing > 0 else "steer_R"
            else:
                skill = "forward"
            if skill != skill0 and skill0 is not None and t_sim - last_switch_t < DWELL_S and skill != "stop":
                skill = skill0
            steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
            act = ACTIONS[skill]
            if skill != skill0:
                last_switch_t = t_sim
                skill0 = skill

            amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
            phases_t += 2 * math.pi * FREQ * DT_CTRL
            duty = np.full(4, DUTY)
            if steer_now > 0:
                duty[list(SIDE["L"])] *= (1 - steer_now * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 + steer_now * STEER_LIMIT)
            elif steer_now < 0:
                duty[list(SIDE["L"])] *= (1 + abs(steer_now) * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 - abs(steer_now) * STEER_LIMIT)
            w = wave_at(phases_t + PHASE0, duty)
            data.ctrl[:] = np.clip(SIGNS * (BIAS + AMPS * amp_scale * w), -CTRL_MAX, CTRL_MAX)

            fell = False
            for _ in range(int(CTRL_HZ // DATA_HZ)):  # 5 控制步 = 1 数据帧
                for _ in range(SUB):
                    mujoco.mj_step(model, data)
                t_sim += DT_CTRL
                if data.body(body_id).xpos[2] < 0.030:
                    fell = True
                    break
            if fell:
                break

            r = data.body(body_id).xmat.reshape(3, 3)
            v_body = r.T @ data.qvel[:3]
            state = np.concatenate([
                np.array([data.qpos[servo_qposadr[i]] for i in range(4)]),
                [float(-v_body[1]), float((r.T @ data.qvel[3:6])[2])],
                [bearing, dist],
            ]).astype(np.float32)
            frames.append({
                "observation.images.camera1": renderer_update(model, data, renderer, "overhead"),
                "observation.images.camera2": renderer_update(model, data, renderer, "front"),
                "observation.images.camera3": renderer_update(model, data, renderer, "oblique"),
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
            print(f"ep{ep:03d} ✓ {color:5s} 目标{np.linalg.norm(tgt_xy):.2f}m → {t_sim:5.1f}s，"
                  f"{len(frames)} 帧（成功 {n_saved}/{args.episodes}，丢弃 {n_fail}）", flush=True)
        else:
            n_fail += 1
            print(f"ep{ep:03d} ✗ {color:5s} 丢弃（{'fallen' if fell else 'timeout'}，{t_sim:.1f}s）", flush=True)

    dataset.finalize()
    print(f"\n完成：保存 {n_saved} / 丢弃 {n_fail}，数据集 {ds_root}")
    os._exit(0)


def renderer_update(model, data, renderer, cam):
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


if __name__ == "__main__":
    main()
