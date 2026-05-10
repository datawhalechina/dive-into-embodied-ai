from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs/practices/quadruped/cs123/figs"
L = (0.3, 0.25, 0.15)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def fk_planar(thetas: np.ndarray, lengths: tuple[float, float, float] = L) -> np.ndarray:
    t1, t2, t3 = thetas
    return (
        rot_z(t1)
        @ trans(lengths[0], 0, 0)
        @ rot_z(t2)
        @ trans(lengths[1], 0, 0)
        @ rot_z(t3)
        @ trans(lengths[2], 0, 0)
    )


def write_workspace_scatter() -> None:
    rng = np.random.default_rng(1234)
    pts = np.empty((30_000, 2))
    for i in range(len(pts)):
        q = rng.uniform(-np.pi, np.pi, size=3)
        pts[i] = fk_planar(q)[:2, 3]

    max_r = sum(L)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.8, alpha=0.18, c="#2563eb", linewidths=0)
    ax.add_patch(
        plt.Circle((0, 0), max_r, fill=False, color="#0f172a", linestyle="--", linewidth=1.2)
    )
    ax.scatter([0], [0], s=18, c="#0f172a", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("3-DoF planar arm reachable workspace")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.text(
        0.32,
        0.66,
        f"outer radius = {max_r:.2f} m",
        fontsize=9,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "planar-workspace-scatter.webp")
    plt.close(fig)


def build_overlay_xml(marker_pos: np.ndarray) -> str:
    x, y, z = marker_pos
    return f"""
<mujoco model="planar_3dof_overlay">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <asset>
    <material name="link_mat" rgba="0.72 0.78 0.86 1"/>
    <material name="joint_mat" rgba="0.12 0.18 0.28 1"/>
    <material name="fk_mat" rgba="1.0 0.05 0.05 1"/>
    <material name="site_mat" rgba="0.1 0.55 1.0 1"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2" dir="0 0 -1"/>
    <camera name="overview" pos="0.38 -1.35 1.05" xyaxes="1 0 0 0 0.62 0.78"/>
    <geom type="plane" size="1.0 1.0 0.01" rgba="0.96 0.97 0.99 1"/>

    <body name="fk_overlay_marker" pos="{x:.8f} {y:.8f} {z:.8f}">
      <geom type="sphere" size="0.025" material="fk_mat"/>
    </body>

    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.025" material="joint_mat"/>
      <geom type="capsule" fromto="0 0 0 {L[0]} 0 0" size="0.016" material="link_mat"/>

      <body name="link2" pos="{L[0]} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.022" material="joint_mat"/>
        <geom type="capsule" fromto="0 0 0 {L[1]} 0 0" size="0.014" material="link_mat"/>

        <body name="link3" pos="{L[1]} 0 0">
          <joint name="j3" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.020" material="joint_mat"/>
          <geom type="capsule" fromto="0 0 0 {L[2]} 0 0" size="0.012" material="link_mat"/>
          <site name="end_site" pos="{L[2]} 0 0" size="0.018" rgba="0.1 0.55 1.0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_mujoco_overlay() -> None:
    q = np.deg2rad([34.0, -62.0, 48.0])
    p = fk_planar(q)[:3, 3]

    model = mujoco.MjModel.from_xml_string(build_overlay_xml(p))
    data = mujoco.MjData(model)
    data.qpos[:3] = q
    mujoco.mj_forward(model, data)

    body_names = ["link1", "link2", "link3"]
    joint_points = [np.zeros(3)]
    for name in body_names[1:]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_points.append(data.xpos[body_id].copy())

    end_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_site")
    end_site = data.site_xpos[end_id].copy()
    points = np.vstack([joint_points, end_site])

    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=180)
    ax.set_facecolor("#101827")
    fig.patch.set_facecolor("white")

    ax.plot(points[:, 0], points[:, 1], color="#cbd5e1", linewidth=18, solid_capstyle="round")
    ax.plot(points[:, 0], points[:, 1], color="#475569", linewidth=2.5)
    ax.scatter(points[:-1, 0], points[:-1, 1], s=95, c="#0f172a", edgecolors="#e2e8f0", linewidths=1.5, zorder=4)
    ax.scatter([end_site[0]], [end_site[1]], s=170, c="#38bdf8", edgecolors="white", linewidths=1.5, zorder=5, label="MuJoCo end_site")
    ax.scatter([p[0]], [p[1]], s=70, c="#ef4444", edgecolors="white", linewidths=1.2, zorder=6, label="Python FK marker")

    ax.annotate(
        "Python FK marker\nmatches end_site",
        xy=(p[0], p[1]),
        xytext=(p[0] + 0.08, p[1] + 0.10),
        arrowprops={"arrowstyle": "->", "color": "#ef4444", "lw": 1.5},
        color="#f8fafc",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#1e293b", "edgecolor": "#475569"},
    )

    ax.set_title("MuJoCo FK check: Python FK marker aligned with end_site", pad=14)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, color="#243244", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.08, 0.78)
    ax.set_ylim(-0.24, 0.52)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mujoco-fk-overlay.webp")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_workspace_scatter()
    write_mujoco_overlay()


if __name__ == "__main__":
    main()
