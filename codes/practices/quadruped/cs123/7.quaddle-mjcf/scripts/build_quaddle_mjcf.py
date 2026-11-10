"""从官方 Quaddle URDF 生成 MuJoCo MJCF 模型。

参考资料：官方网页仿真器 `one-for-all/quaddle-sim` 的 builder.rs。
机器人是 4 条“平行曲柄 + 闭链”腿：每条腿 1 个主动舵机关节（thigh）+
5 个被动关节（motor_arm / spring / leg / wheel / wheel_tip），thigh 与
leg 之间用两点 `connect` 约束闭合，等效一个转动副。

生成约定：
- 所有关节 `ref` 设为解算出的装配角，因此 **q=0 恰好是官方上电站立姿态**
  （固件角度映射：LF=0°, RF=0°, RB=+40°, LB=-40°，URDF 轴向约定）。
- 闭链锚点以 site 对（thigh 侧 / leg 侧）给定，装配位姿下天然重合。
- 碰撞几何完全照抄官方 builder.rs：body 一个 box、每腿 thigh box、
  motor_arm box、leg 两个球（foot 3.5mm + knee 2.5mm）、wheel_tip 球 2mm。
- 视觉 mesh 按白名单内嵌（跳过 53MB 的 spring STL），并按装配旋转
  重新表达到各自 body 的新局部系。

用法：
    uv run python 7.quaddle-mjcf/scripts/build_quaddle_mjcf.py
输出：
    7.quaddle-mjcf/quaddle.xml
    7.quaddle-mjcf/scripts/closure_report.json
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent.parent
URDF_PATH = LAB_DIR / "assets" / "quaddle_urdf" / "robot.urdf"
OUT_XML = LAB_DIR / "quaddle.xml"
OUT_REPORT = LAB_DIR / "scripts" / "closure_report.json"

# ---------------------------------------------------------------------------
# 常量（来自官方 quaddle-sim builder.rs / OpenCat 固件）
# ---------------------------------------------------------------------------

LEGS = ["lf", "rf", "rb", "lb"]  # 固件关节顺序 2..5：LF, RF, RB, LB
LEG_FULL = {
    "lf": "left_front",
    "rf": "right_front",
    "rb": "right_back",
    "lb": "left_back",
}
# 固件 power_on_angles（builder.rs）：LF=0, RF=0, RB=40, LB=40（度）
POWER_ON_DEG = {"lf": 0.0, "rf": 0.0, "rb": 40.0, "lb": 40.0}
# 固件 rotation_directions：LF +1, RF -1, RB -1, LB +1
ROT_DIR = {"lf": 1.0, "rf": -1.0, "rb": -1.0, "lb": 1.0}
# 官方 spawn 时 thigh 与 motor_arm 刚性同转：URDF 轴向与官方引擎轴向相反
THIGH_HOME_RAD = {
    k: -ROT_DIR[k] * math.radians(POWER_ON_DEG[k]) for k in LEGS
}

# 碰撞几何（builder.rs）
BODY_BOX_WD = (0.0498, 0.105, 0.027)
THIGH_BOX = (0.017, 0.0436, 0.008)
THIGH_BOX_Y = {"left": 0.0133, "right": -0.0133}
MOTOR_ARM_BOX = (0.02, 0.011, 0.0065)
LEG_FOOT_SPHERE_R = 0.0035
LEG_KNEE_SPHERE_R = 0.0025
WHEEL_TIP_SPHERE_R = 0.002

# 闭链锚点：沿铰链轴偏移的两点，等效转动副
CLOSURE_AXIS_OFFSET = 0.006

# 舵机参数（估）：P1L 总成 kp / 阻尼 / 力矩限幅；转子惯量加到 motor_arm
SERVO_KP = 3.0
SERVO_KV = 0.02
SERVO_FORCE = 0.30
SERVO_CTRL_RANGE = math.radians(140.0)
MOTOR_ARM_ARMATURE = 2.0e-5
MOTOR_ARM_FRICTION = 0.02

SPRING_STIFFNESS = 0.6  # 扭簧刚度（N m/rad）：0.15 太软塌陷，2.0 在 dt=2ms 数值不稳定

# 视觉 mesh 白名单（跳过 53MB 的 spring STL）
VISUAL_WHITELIST = {
    "body", "body_bar", "gear", "lego_top_cap", "power_cap",
    "battery_cap_thin", "batterie_model_bl_5c", "screw_servo",
    "thigh", "thigh_right", "leg", "leg_right", "motor_arm",
    "lower4wheel", "tip_wheel", "bearing",
}

# ---------------------------------------------------------------------------
# 小型旋转工具
# ---------------------------------------------------------------------------


def rpy_to_r(rpy) -> np.ndarray:
    """URDF rpy（绕固定轴 X->Y->Z 外旋）-> 旋转矩阵 R = Rz(yaw)Ry(pit)Rx(rol)。"""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def r_to_quat(r: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 (w, x, y, z)。数值稳定。"""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] >= r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def se3_from(xyz, R) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(xyz, dtype=float), np.asarray(R, dtype=float)


def se3_mul(p1, R1, p2, R2):
    """(p1,R1)∘(p2,R2)：p2/R2 在 1 坐标系中 -> 返回在 1 的父系中的位姿。"""
    return p1 + R1 @ p2, R1 @ R2


def fmt(vals) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.atleast_1d(vals))


# ---------------------------------------------------------------------------
# URDF 解析
# ---------------------------------------------------------------------------


class UrdfModel:
    def __init__(self, path: Path):
        self.root = ET.parse(path).getroot()
        self.links: dict[str, dict] = {}
        for link in self.root.findall("link"):
            name = link.get("name")
            info: dict = {"inertial": None, "visuals": []}
            ine = link.find("inertial")
            if ine is not None:
                o = ine.find("origin")
                xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                mass = float(ine.find("mass").get("value"))
                i = ine.find("inertia")
                I = np.array(
                    [
                        [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                        [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                        [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
                    ]
                )
                info["inertial"] = {"xyz": xyz, "rpy": rpy, "mass": mass, "I": I}
            for vis in link.findall("visual"):
                o = vis.find("origin")
                vxyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                vrpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                mesh = vis.find("geometry/mesh")
                fname = mesh.get("filename") if mesh is not None else None
                info["visuals"].append({"xyz": vxyz, "rpy": vrpy, "mesh": fname})
            self.links[name] = info
        self.joints: dict[str, dict] = {}
        for j in self.root.findall("joint"):
            name = j.get("name")
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            info = {
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
            self.joints[name] = info

    def joint(self, name: str) -> dict:
        return self.joints[f"{name}_frame" if f"{name}_frame" in self.joints else name]


# ---------------------------------------------------------------------------
# 闭链求解（每条腿 3 个被动关节角 ψ(motor_arm), σ(spring), λ(leg)）
# ---------------------------------------------------------------------------


class LegChain:
    """一条腿的运动链，全部坐标都在 body 系内。"""

    def __init__(self, urdf: UrdfModel, side: str, direction: str, short: str):
        full = f"{side}_{direction}"
        self.side, self.direction, self.short = side, direction, short

        j_thigh = urdf.joints[f"{full}_thigh"]
        self.A1, self.R_th0 = se3_from(j_thigh["xyz"], rpy_to_r(j_thigh["rpy"]))
        self.c2 = np.array(urdf.joints[f"closing_{full}_leg_2_frame"]["xyz"])

        j_ma = urdf.joints[f"{full}_motor_arm"]
        self.A2, self.R_ma0 = se3_from(j_ma["xyz"], rpy_to_r(j_ma["rpy"]))

        j_spring = urdf.joints[f"{full}_spring"]
        self.s_org = np.array(j_spring["xyz"])
        self.R_spring0 = rpy_to_r(j_spring["rpy"])

        j_leg = urdf.joints[f"{full}_leg"]
        self.l_org = np.array(j_leg["xyz"])
        self.R_leg0 = rpy_to_r(j_leg["rpy"])

        self.c1 = np.array(urdf.joints[f"closing_{full}_leg_1_frame"]["xyz"])
        foot = urdf.joints[f"{full}_foot_frame"]["xyz"]
        self.foot_local = np.array(foot)
        wheel = urdf.joints[f"{full}_wheel"]["xyz"]
        self.wheel_local = np.array(wheel)

    # -- 正运动学（θ: thigh 绝对角，其他为绝对角；返回关键点） ----------------

    def fk(self, theta, psi, sigma, lam):
        R_th = self.R_th0 @ rz(theta)
        B = self.A1 + R_th @ self.c2

        R_ma = self.R_ma0 @ rz(psi)
        p_spring, R_spring_joint = se3_mul(self.A2, R_ma, self.s_org, self.R_spring0)
        R_sj = R_spring_joint @ rz(sigma)
        p_leg, R_leg_joint = se3_mul(p_spring, R_sj, self.l_org, self.R_leg0)
        R_l = R_leg_joint @ rz(lam)
        C = p_leg + R_l @ self.c1

        foot = p_leg + R_l @ self.foot_local
        wheel = p_leg + R_l @ self.wheel_local
        return {
            "B": B, "C": C, "err": C - B,
            "p_spring": p_spring, "p_leg": p_leg,
            "R_th": R_th, "R_ma": R_ma, "R_sj": R_sj, "R_l": R_l,
            "foot": foot, "wheel": wheel,
        }

    # -- 装配解（解析） -------------------------------------------------------
    # 官方 quaddle-sim 的结论：thigh 与 motor_arm 是平行曲柄，σ=λ=0 时整条腿
    # 刚性随动（"turning both by the same amount turns the whole leg linkage
    # rigidly"）。URDF 零位在 y-z 平面内天然闭合，仅 x 向留有 ~2mm CAD 间隙
    # （绕 body-x 的转动不改变 x 坐标，间隙由闭链约束焊接吸收）。

    def solve(self, theta: float) -> tuple[float, float, float, float, float]:
        psi, sigma, lam = theta, 0.0, 0.0
        err = self.fk(theta, psi, sigma, lam)["err"]
        yz = float(math.hypot(err[1], err[2]))
        if yz > 1e-4:
            raise RuntimeError(
                f"[{self.short}] URDF 零位 y-z 平面不闭合（残差 {yz*1000:.2f}mm），"
                "URDF 或轴系解析有误"
            )
        return psi, sigma, lam, float(err[0]), yz


# ---------------------------------------------------------------------------
# MJCF 生成
# ---------------------------------------------------------------------------

MESH_REL = "assets/quaddle_urdf/assets"


def add_inertial(elem: ET.SubElement, inertial: dict | None, Rz_q: np.ndarray, label: str):
    """URDF inertial -> MJCF <inertial>，并把位姿/张量旋转到装配后 body 系。"""
    if inertial is None:
        raise RuntimeError(f"{label}: URDF link 缺少 inertial，请检查 URDF 完整性")
    R = Rz_q  # 装配旋转（body 系 = URDF-zero 系绕自身 z 旋转后）
    p_new = R @ np.array(inertial["xyz"])
    I0 = inertial["I"]
    I_new = R @ I0 @ R.T
    I_new = 0.5 * (I_new + I_new.T)
    e = ET.SubElement(
        elem, "inertial",
        pos=fmt(p_new), mass=f"{inertial['mass']:.9g}",
        fullinertia=fmt([I_new[0, 0], I_new[1, 1], I_new[2, 2],
                         I_new[0, 1], I_new[0, 2], I_new[1, 2]]),
    )
    return e


def add_visuals(elem, visuals: list[dict], Rz_q: np.ndarray):
    R = Rz_q
    for vis in visuals:
        mesh = vis["mesh"]
        if mesh is None:
            continue
        fname = Path(mesh).name
        if fname.replace(".stl", "") not in VISUAL_WHITELIST:
            continue
        pos = R @ np.array(vis["xyz"])
        quat_v = r_to_quat(R @ rpy_to_r(vis["rpy"]))
        ET.SubElement(
            elem, "geom",
            type="mesh", mesh=fname.replace(".stl", ""),
            pos=fmt(pos), quat=fmt(quat_v),
            contype="0", conaffinity="0", group="1",
        )


def main():
    urdf = UrdfModel(URDF_PATH)
    root = ET.Element("mujoco", model="quaddle")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(
        root, "option",
        timestep="0.001", integrator="implicitfast",
        cone="elliptic", solver="Newton",
        iterations="100", tolerance="1e-10",
    )

    asset = ET.SubElement(root, "asset")
    seen = set()
    for link in urdf.links.values():
        for vis in link["visuals"]:
            if vis["mesh"] is None:
                continue
            fname = Path(vis["mesh"]).name
            if fname.replace(".stl", "") not in VISUAL_WHITELIST or fname in seen:
                continue
            seen.add(fname)
            ET.SubElement(asset, "mesh", name=fname.replace(".stl", ""),
                          file=f"{MESH_REL}/{fname}")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.2 0.3 0.4", rgb2="0.3 0.4 0.5", width="300", height="300")
    ET.SubElement(asset, "material", name="grid", texture="grid",
                  texrepeat="8 8", reflectance="0.1")

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction="1.0 0.02 0.001",
                  condim="3", solref="0.01 1", solimp="0.9 0.95 0.001")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.6 0.3 1.2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="8 8 0.1",
                  material="grid", friction="1.0")
    # 固定跟踪位相机（W2 采集时再升级为跟随相机）
    cam = ET.SubElement(world, "camera", name="tracking",
                        pos="0.5 -0.5 0.35", xyaxes="0.707 0.707 0 -0.273 0.273 0.923")

    # 根 body：URDF body 系，旋转 +90° yaw 使“前进(-Y_urdf)”指向世界 +X
    spawn_z = 0.06
    yaw_q = r_to_quat(rz(math.pi / 2))
    root_body = ET.SubElement(
        world, "body", name="body", pos=f"0 0 {spawn_z}", quat=fmt(yaw_q)
    )
    ET.SubElement(root_body, "freejoint", name="root")
    add_inertial(root_body, urdf.links["body"]["inertial"], np.eye(3), "body")
    add_visuals(root_body, urdf.links["body"]["visuals"], np.eye(3))
    # body 碰撞 box（builder.rs：body_collision_frame 处，向下 h）
    p_bc = np.array(urdf.joints["body_collision_frame"]["xyz"])
    h = BODY_BOX_WD[2]
    ET.SubElement(root_body, "geom", name="body_collision", type="box",
                  pos=fmt(p_bc - np.array([0, 0, h / 2])),
                  size=fmt([BODY_BOX_WD[0] / 2, BODY_BOX_WD[1] / 2, h / 2]))

    report = {"legs": {}, "spawn_z": spawn_z}
    eqs = ET.SubElement(root, "equality")
    contact = ET.SubElement(root, "contact")
    actuators = ET.SubElement(root, "actuator")

    # ---- 平衡站姿搜索：保持官方 rest 的后腿 ±40°，扫描前腿角使四足等高 -------
    lb_chain = LegChain(urdf, "left", "back", "lb")
    lf_chain = LegChain(urdf, "left", "front", "lf")
    back_foot_z = lb_chain.fk(THIGH_HOME_RAD["lb"], THIGH_HOME_RAD["lb"], 0, 0)["foot"][2]
    grid = np.linspace(math.radians(-20), math.radians(20), 801)
    front_err = [
        abs(lf_chain.fk(t, t, 0, 0)["foot"][2] - back_foot_z) for t in grid
    ]
    front_adj = float(grid[int(np.argmin(front_err))])
    if front_err[int(np.argmin(front_err))] > 1e-4:
        print(f"警告：前腿等高补偿残差 {front_err[int(np.argmin(front_err))]*1000:.2f}mm")
    THIGH_HOME_RAD["lf"] = THIGH_HOME_RAD["rf"] = front_adj
    report["balanced_stand"] = {
        "back_legs_deg": 40.0,
        "front_legs_adj_deg": math.degrees(front_adj),
        "front_back_foot_z_diff_m": float(min(front_err)),
    }

    leg_bodies = []
    solved = {}

    for short in LEGS:
        side = "left" if short in ("lf", "lb") else "right"
        direction = "front" if short in ("lf", "rf") else "back"
        full = LEG_FULL[short]
        chain = LegChain(urdf, side, direction, short)
        theta = THIGH_HOME_RAD[short]
        psi, sigma, lam, err_x, yz = chain.solve(theta)
        fk = chain.fk(theta, psi, sigma, lam)
        solved[short] = (theta, psi, sigma, lam, fk)

        report["legs"][short] = {
            "thigh_home_deg": math.degrees(theta),
            "psi_deg": math.degrees(psi),
            "sigma_deg": math.degrees(sigma),
            "lam_deg": math.degrees(lam),
            "closure_residual_x_m": err_x,
            "closure_residual_yz_m": yz,
            "foot_body_frame": [float(v) for v in fk["foot"]],
            "wheel_body_frame": [float(v) for v in fk["wheel"]],
        }

        def E(tag, parent, **attrs):
            return ET.SubElement(
                parent, tag,
                **{k: fmt(v) if isinstance(v, (float, int, list, tuple, np.ndarray)) else v
                   for k, v in attrs.items()},
            )

        # ---- thigh（主动） ----------------------------------------------------
        # 装配位姿：joint origin 平移不变，姿态 = R_th0 · Rz(theta)
        p_th, R_th_a = chain.A1, chain.R_th0 @ rz(theta)
        b_th = E("body", root_body, name=f"{short}_thigh", pos=p_th, quat=r_to_quat(R_th_a))
        E("joint", b_th, name=f"{short}_thigh", type="hinge", axis="0 0 1",
          ref=theta, damping="1e-4", frictionloss="0.005")
        add_inertial(b_th, urdf.links[f"{full}_thigh"]["inertial"], rz(theta), f"{short}_thigh")
        add_visuals(b_th, urdf.links[f"{full}_thigh"]["visuals"], rz(theta))
        E("geom", b_th, name=f"{short}_thigh_col", type="box",
          pos=[0.0, THIGH_BOX_Y[side], -0.0015],
          size=[THIGH_BOX[0] / 2, THIGH_BOX[1] / 2, THIGH_BOX[2] / 2])

        # ---- motor_arm（被动，带舵机转子 armature） ----------------------------
        p_ma, R_ma_a = chain.A2, chain.R_ma0 @ rz(psi)
        b_ma = E("body", root_body, name=f"{short}_motor_arm", pos=p_ma, quat=r_to_quat(R_ma_a))
        E("joint", b_ma, name=f"{short}_motor_arm", type="hinge", axis="0 0 1",
          ref=psi, armature=MOTOR_ARM_ARMATURE, frictionloss=MOTOR_ARM_FRICTION)
        add_inertial(b_ma, urdf.links[f"{full}_motor_arm"]["inertial"], rz(psi), f"{short}_motor_arm")
        add_visuals(b_ma, urdf.links[f"{full}_motor_arm"]["visuals"], rz(psi))
        E("geom", b_ma, name=f"{short}_ma_col", type="box",
          pos=[0.0045, 0, 0.00055],
          size=[MOTOR_ARM_BOX[0] / 2, MOTOR_ARM_BOX[1] / 2, MOTOR_ARM_BOX[2] / 2])

        # ---- spring（被动扭簧） ------------------------------------------------
        # FK：p_spring/R_sj 是装配姿态（ motor_arm 装配系中表达）
        fk_ps = fk["p_spring"]
        R_sj_a = fk["R_sj"]
        # motor_arm 装配系中的局部位姿：先转到 body 系再回到 motor_arm 装配系
        p_s_local = R_ma_a.T @ (fk_ps - p_ma)
        R_s_local = R_ma_a.T @ R_sj_a
        b_spr = E("body", root_body, name=f"{short}_spring", pos=p_s_local, quat=r_to_quat(R_s_local))
        # spring 的父 body 是 motor_arm：MJCF 里作为 motor_arm 的子 body
        root_body.remove(b_spr)
        b_ma.append(b_spr)
        E("joint", b_spr, name=f"{short}_spring", type="hinge", axis="0 0 1",
          ref=sigma, stiffness=SPRING_STIFFNESS, springref=sigma,
          damping="1e-3", range="-0.3 0.3", limited="true")
        add_inertial(b_spr, urdf.links[f"{full}_spring"]["inertial"], R_sj_a, f"{short}_spring")

        # ---- leg（被动） -------------------------------------------------------
        p_leg_body, R_leg_a = fk["p_leg"], fk["R_l"]
        p_l_local = R_sj_a.T @ (p_leg_body - fk_ps)
        R_l_local = R_sj_a.T @ R_leg_a
        b_leg = E("body", root_body, name=f"{short}_leg", pos=p_l_local, quat=r_to_quat(R_l_local))
        root_body.remove(b_leg)
        b_spr.append(b_leg)
        E("joint", b_leg, name=f"{short}_leg", type="hinge", axis="0 0 1",
          ref=lam, damping="1e-3")
        add_inertial(b_leg, urdf.links[f"{full}_leg"]["inertial"], R_leg_a, f"{short}_leg")
        add_visuals(b_leg, urdf.links[f"{full}_leg"]["visuals"], R_leg_a)
        # 腿板防穿模球：纯视觉占位（contype=0），足端接触统一交给滚轮——
        # 球足端的各向同性滑动摩擦曾把稳定行走速度锁死在 ~10mm/s（真机靠滚轮）
        E("geom", b_leg, name=f"{short}_foot_col", type="sphere",
          pos=chain.foot_local, size=[LEG_FOOT_SPHERE_R],
          contype="0", conaffinity="0")
        E("geom", b_leg, name=f"{short}_knee_col", type="sphere",
          pos=[0, 0, 0], size=[LEG_KNEE_SPHERE_R],
          contype="0", conaffinity="0")

        # ---- wheel + wheel_tip（被动滚转，视觉为主） ----------------------------
        wheel_full = urdf.joints[f"{full}_wheel"]
        R_w_joint = R_leg_a @ rpy_to_r(wheel_full["rpy"])
        p_w_world = p_leg_body + R_leg_a @ np.array(wheel_full["xyz"])
        p_w_local = R_leg_a.T @ (p_w_world - p_leg_body)
        b_w = E("body", root_body, name=f"{short}_wheel", pos=p_w_local,
                quat=r_to_quat(R_leg_a.T @ R_w_joint))
        root_body.remove(b_w)
        b_leg.append(b_w)
        w_rpy = rpy_to_r(wheel_full["rpy"])
        E("joint", b_w, name=f"{short}_wheel", type="hinge", axis="0 0 1",
          damping="1e-4")
        add_inertial(b_w, urdf.links[f"{full}_wheel"]["inertial"], R_w_joint, f"{short}_wheel")
        add_visuals(b_w, urdf.links[f"{full}_wheel"]["visuals"], R_w_joint)

        tip_full = urdf.joints[f"{full}_wheel_tip"]
        R_t_joint = R_w_joint @ rpy_to_r(tip_full["rpy"])
        p_t_world = p_w_world + R_w_joint @ np.array(tip_full["xyz"])
        p_t_local = R_w_joint.T @ (p_t_world - p_w_world)
        b_t = E("body", root_body, name=f"{short}_wheel_tip", pos=p_t_local,
                quat=r_to_quat(R_w_joint.T @ R_t_joint))
        root_body.remove(b_t)
        b_w.append(b_t)
        E("joint", b_t, name=f"{short}_wheel_tip", type="hinge", axis="0 0 1",
          damping="1e-4")
        add_inertial(b_t, urdf.links[f"{full}_wheel_tip"]["inertial"], R_t_joint, f"{short}_wheel_tip")
        add_visuals(b_t, urdf.links[f"{full}_wheel_tip"]["visuals"], R_t_joint)
        tip_col_frame = urdf.joints[f"{full}_wheel_tip_collision_frame"]["xyz"]
        # 滚轮足端：圆柱（r 3.5mm × 宽 4mm），轴 = wheel_tip 铰链轴（local z）。
        # 轮缘触地后身体前移会驱动轮滚（关节自由度），滚动方向低阻力、
        # 轮轴方向被铰链锁定 —— 这是真机 omni 轮足端的行走机制来源。
        E("geom", b_t, name=f"{short}_tip_col", type="cylinder",
          pos=tip_col_frame, size=[0.0035, 0.002])

        # ---- 闭链 site 对 + 两个 connect --------------------------------------
        # thigh 侧锚点 = closing_2 原点（thigh 局部系），沿 thigh 局部 z 偏移两点
        a_local = chain.c2
        b_local = chain.c2 + np.array([0, 0, CLOSURE_AXIS_OFFSET])
        # leg 侧对应点：body 系中 = B 与 B + R_l·(R_th·z)·offset，转到 leg 局部
        axis_body = R_th_a @ np.array([0, 0, 1.0])
        a_world_leg = fk["B"]
        b_world_leg = fk["B"] + axis_body * CLOSURE_AXIS_OFFSET
        a_leg_local = R_leg_a.T @ (a_world_leg - p_leg_body)
        b_leg_local = R_leg_a.T @ (b_world_leg - p_leg_body)
        for tag, p_th_loc, p_lg_loc in (("a", a_local, a_leg_local), ("b", b_local, b_leg_local)):
            ET.SubElement(b_th, "site", name=f"{short}_closure_thigh_{tag}",
                          pos=fmt(p_th_loc), size="0.002", rgba="1 0 0 1")
            ET.SubElement(b_leg, "site", name=f"{short}_closure_leg_{tag}",
                          pos=fmt(p_lg_loc), size="0.002", rgba="0 0 1 1")
            E("connect", eqs, name=f"{short}_closure_{tag}",
              site1=f"{short}_closure_thigh_{tag}", site2=f"{short}_closure_leg_{tag}",
              solref="0.002 1", solimp="0.95 0.99 0.001")

        # 同腿内部碰撞全部排除（闭链构件物理上同轴嵌套）
        within = [f"{short}_thigh", f"{short}_motor_arm", f"{short}_spring",
                  f"{short}_leg", f"{short}_wheel", f"{short}_wheel_tip"]
        for i in range(len(within)):
            for j in range(i + 1, len(within)):
                E("exclude", contact, body1=within[i], body2=within[j])
        for b in within:
            E("exclude", contact, body1="body", body2=b)

        # ---- 舵机 --------------------------------------------------------------
        E("position", actuators, name=f"{short}_servo", joint=f"{short}_thigh",
          kp=SERVO_KP, kv=SERVO_KV, ctrlrange=[-SERVO_CTRL_RANGE, SERVO_CTRL_RANGE],
          forcerange=[-SERVO_FORCE, SERVO_FORCE])

        leg_bodies.append(short)

    # ---- keyframe：站立（q 全 0）+ spawn 高度 ------------------------------
    kf = ET.SubElement(root, "keyframe")
    qpos = [0, 0, spawn_z] + list(yaw_q) + [0.0] * (4 * 6)
    ET.SubElement(kf, "key", name="home", qpos=fmt(qpos),
                  ctrl=fmt([0.0] * 4))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(OUT_XML, encoding="unicode", xml_declaration=False)

    total_mass = 0.0
    for link in urdf.links.values():
        if link["inertial"]:
            total_mass += link["inertial"]["mass"]
    report["total_mass_kg"] = round(total_mass, 6)
    report["meshes"] = sorted(seen)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"写入 {OUT_XML}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
