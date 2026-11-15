"""v4 前置验证：专家闭环下 spin（支点转）原语 vs v3 弧线专家（rear 场景对比）。

W5 根因：v3 专家只有差频弧线转向，目标近正后方（|bearing|>120°）时弧线稳态
bearing 恒定 → 死锁。本脚本把 spin_feasibility.py 扫出的可行支点转
（pivot_*_a0：一侧腿停摆、另一侧 trot，wz≈±0.4-0.5 rad/s，连续 10s 不翻）
接入方位闭环专家，在"目标固定正后方"的场景里对比两种专家：

    arc_only : v3 现行专家（死区 ±25°，弧线转向）——预期死锁超时
    arc_spin : |bearing|>120° 先支点转到 |bearing|<60°，再走 v3 逻辑——预期到达

判定标准：arc_spin 3/3 到达且 arc_only 死锁 → spin 原语闭环有效，进入
collect.py v4 改造（3 维动作 + bearing 过采样）。

运行（仅需 mujoco==3.8.1 + numpy，无需 lerobot/渲染）：
    python expert_spin_smoke.py [--scenes 3] [--seed 4100]
"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CS123 = HERE.parent.parent / "quadruped" / "cs123"
sys.path.insert(0, str(CS123 / "8.rl_quaddle"))
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402

from collect import (  # noqa: E402
    BEARING_DEADZONE, COLORS, OBJ_RNG, SUCCESS_DIST, TIMEOUT_S,
    build_scene, random_layout, yaw_quat,
)

SETTLE_S = 1.2
SPIN_ENTER = math.radians(50.0)    # （保留常量供 v3 对照参考；v4 转向域=死区）
ARC_MAX = 0.55                     # （v4 已弃用弧线档；常量仅存档）
DWELL_S = 0.4                      # 切档最小驻留：防 amp 斜坡反复清零的抖动冲击

# spin_feasibility.py 扫出的可行支点转（腿序 lf, rf, rb, lb）：
# spin_left  = 左腿(lf,lb)停摆 + 右腿(rf,rb) trot → wz +0.535 rad/s
# spin_right = 镜像 → wz −0.418 rad/s
TROT = (0.0, math.pi / 2, 0.0, math.pi / 2)
PIVOT_SKILLS = {
    "spin_left": [
        (0.0, 2.5, TROT[0], -0.45), (10.5, 2.5, TROT[1], -0.45),
        (10.5, 2.5, TROT[2], -0.45), (0.0, 2.5, TROT[3], -0.45),
    ],
    "spin_right": [
        (10.5, 2.5, TROT[0], -0.45), (0.0, 2.5, TROT[1], -0.45),
        (0.0, 2.5, TROT[2], -0.45), (10.5, 2.5, TROT[3], -0.45),
    ],
}


def make_ctrl(model, data):
    """与 collect.py 相同的 __new__ 构造（物理状态由外部 reset 管理）。"""
    from cpg_controller import CpgController
    ctrl = CpgController.__new__(CpgController)
    ctrl.env = types.SimpleNamespace(n_substeps=20)
    ctrl.model, ctrl.data = model, data
    ctrl.dt = 0.02
    ctrl._phases = np.zeros(4)
    ctrl._amp_scale, ctrl._dir = 0.0, 1.0
    ctrl.skill = "stop"
    return ctrl


def run_episode(model, data, body_id, geom_ids, layout, target_xy, yaw0, use_spin,
                deadzone: float = BEARING_DEADZONE):
    import cpg_controller as cpg_mod
    if use_spin:
        cpg_mod.SKILLS.update(PIVOT_SKILLS)  # 注入支点转档（模块级表）

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[3:7] = yaw_quat(yaw0)
    for i, c in enumerate(COLORS):
        model.geom_pos[geom_ids[c]][:2] = list(layout[i])
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, data)

    ctrl = make_ctrl(model, data)
    tgt = np.array([target_xy[0], target_xy[1], 0.0])
    t, skill0 = 0.0, None
    last_switch_t = 0.0
    spin_active, spin_dir = False, None
    success = fallen = False
    trace = []
    while t < TIMEOUT_S:
        r = data.body(body_id).xmat.reshape(3, 3)
        pos = data.body(body_id).xpos
        d_world = tgt - pos
        dist = float(np.linalg.norm(d_world[:2]))
        d_body = r.T @ d_world
        bearing = math.atan2(d_body[0], -d_body[1])  # 左正

        if dist < SUCCESS_DIST:
            skill = "stop"
            success = True
        elif use_spin and abs(bearing) > deadzone:
            # v4：转向 100% 用支点转（weak/strong 弧线档均不可靠：weak 净转向≈0
            # 且符号随机——W2 只标定了 strong 表；strong tier 2s 内翻倒）。
            # 方向锁定：近 ±180° 时符号会被噪声翻转
            if not spin_active:
                spin_active, spin_dir = True, ("spin_left" if bearing > 0 else "spin_right")
            skill = spin_dir
        else:
            spin_active = False
            if not use_spin and abs(bearing) > deadzone:
                skill = "turn_left" if bearing > 0 else "turn_right"
            else:
                skill = "forward"

        # v4：切档最小驻留 0.4s（死区边界 spin↔forward 抖动会让 amp 斜坡反复
        # 清零重爬，腿部冲击是近目标翻倒的诱因）；stop 与对照组不受限
        if (use_spin and skill != skill0 and skill0 is not None
                and t - last_switch_t < DWELL_S and skill != "stop"):
            skill = skill0
        if skill.startswith("turn"):
            ctrl.set_turn_skill(skill.split("_")[1], 1.0)  # v3 对照组：strong tier
        else:
            ctrl.set_skill(skill)
        if skill != skill0:
            trace.append((t, math.degrees(bearing), dist, skill))
            skill0 = skill
            last_switch_t = t

        for _ in range(5):  # 0.1s
            ctrl.step()
            t += ctrl.dt
            if data.body(body_id).xpos[2] < 0.030:
                fallen = True
                break
        if fallen:
            break
        if success:  # 到达即止（与 collect.py 语义一致）
            break
    return {
        "success": success, "fallen": fallen, "t": t,
        "dist_final": dist, "bearing_final_deg": math.degrees(bearing),
        "switches": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4100)
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    scene_xml = out_dir / "scene.xml"
    build_scene(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {
        c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
        for c in COLORS
    }

    rng = np.random.default_rng(args.seed)
    hdr = (f"{'scene':<6} {'expert':<10} {'结果':<6} {'用时':>6} {'末距(cm)':>8} "
           f"{'末bearing°':>10}  档位切换序列（t, bearing°, dist, skill）")
    print(hdr)
    print("-" * len(hdr))
    tally: dict[str, list[str]] = {}
    for si in range(args.scenes):
        layout = random_layout(rng)
        color = COLORS[si % 3]
        target_xy = layout[COLORS.index(color)]
        th_t = math.atan2(target_xy[1], target_xy[0])
        # 前 2/3 场景：目标正后方（home 前进=(sin yaw0,−cos yaw0)，与目标方位差
        # −π/2 即背对）；末 1/3：前侧方（±60° 内）——验证 v4 专家不回退
        if si < args.scenes * 2 // 3:
            yaw0 = th_t - math.pi / 2 + rng.uniform(-math.radians(15), math.radians(15))
        else:
            yaw0 = th_t + rng.uniform(-math.radians(60), math.radians(60))
        for expert in ("v3_arc", "v4_spin"):
            m = run_episode(model, data, body_id, geom_ids, layout, target_xy, yaw0,
                            use_spin=(expert == "v4_spin"))
            tag = "✓到达" if m["success"] else ("✗翻倒" if m["fallen"] else "✗超时")
            tally.setdefault(expert, []).append(tag)
            seq = " ".join(f"[{t:.1f}s {b:+.0f}° {d:.2f}m {s}]"
                           for t, b, d, s in m["switches"][:8])
            print(f"{si:<6} {expert:<10} {tag:<6} {m['t']:>6.1f} "
                  f"{m['dist_final']*100:>8.1f} {m['bearing_final_deg']:>10.1f}  {seq}",
                  flush=True)
    print("-" * len(hdr))
    for k, v in tally.items():
        ok = sum(1 for x in v if "✓" in x)
        n_fall = sum(1 for x in v if "翻" in x)
        n_to = sum(1 for x in v if "超" in x)
        print(f"{k}: 到达 {ok}/{len(v)}  翻倒 {n_fall}  超时 {n_to}")


if __name__ == "__main__":
    main()
