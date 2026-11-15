# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=1.26", "pillow>=10"]
# ///
"""Regenerate Figure 7 with `uv run scripts/generate-cs123-planar-fk.py`.

Joint angles are signed rotations from the preceding link's forward direction.
All extensions and arc endpoints are computed from the same FK geometry.
The figure uses mathematical notation so every locale can share the asset.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/practices/quadruped/cs123/figs/planar-3dof-fk.webp"
)
LENGTHS = np.array([0.30, 0.25, 0.15])
JOINT_ANGLES = np.deg2rad([30.0, 25.0, -25.0])
COLORS = ("#2563eb", "#15803d", "#d97706")
INK = "#0f172a"


def draw_figure() -> None:
    directions = np.cumsum(JOINT_ANGLES)
    vectors = LENGTHS[:, None] * np.column_stack(
        (np.cos(directions), np.sin(directions))
    )
    points = np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))
    end_x, end_y = points[-1]

    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}):
        fig = plt.figure(figsize=(12, 7.8), dpi=180, facecolor="white")
        ax = fig.add_axes((0.055, 0.29, 0.61, 0.62))
        ax.set_aspect("equal")
        ax.set_xlim(-0.045, 0.67)
        ax.set_ylim(-0.06, 0.56)
        ax.axis("off")
        fig.text(0.065, 0.94, "3-DoF", fontsize=23, weight="bold", color=INK)

        for tip, label, offset in [
            ((0.65, 0), "$x$", (0.012, -0.005)),
            ((0, 0.54), "$y$", (-0.025, 0.01)),
        ]:
            ax.annotate("", xy=tip, xytext=(0, 0), arrowprops={
                "arrowstyle": "-|>", "color": INK, "lw": 1.8,
            })
            ax.text(tip[0] + offset[0], tip[1] + offset[1], label, fontsize=17)

        for x, y in points[1:]:
            ax.plot([x, x], [0, y], "--", color="#cbd5e1", lw=1, zorder=0)
            ax.plot([0, x], [y, y], "--", color="#cbd5e1", lw=1, zorder=0)

        for i, (start, end, color) in enumerate(zip(points[:-1], points[1:], COLORS)):
            # The previous link continues FORWARD through this joint; it does
            # not point back toward the previous joint or along the global x axis.
            reference = 0.0 if i == 0 else directions[i - 1]
            extension = start + 0.14 * np.array([np.cos(reference), np.sin(reference)])
            ax.plot(*np.vstack((start, extension)).T, "--", color="#64748b", lw=1.5)
            ax.plot(*np.vstack((start, end)).T, color=color, lw=6, solid_capstyle="round")

            midpoint = (start + end) / 2
            normal = np.array([-np.sin(directions[i]), np.cos(directions[i])])
            # Keep L3 below its link, clear of the clockwise angle annotation.
            label = midpoint + (-0.035 if i == 2 else 0.029) * normal
            ax.text(*label, f"$L_{i + 1}$", color=color, fontsize=19, ha="center", va="center")

            # Sampling in signed order gives a CCW arrow for theta > 0 and
            # a clockwise arrow for theta < 0, including the third joint.
            angles = np.linspace(reference, directions[i], 64)
            arc = start + 0.075 * np.column_stack((np.cos(angles), np.sin(angles)))
            ax.add_patch(FancyArrowPatch(
                path=MplPath(arc), arrowstyle="-|>", mutation_scale=12,
                lw=1.6, color=INK, zorder=4,
            ))
            middle_angle = (reference + directions[i]) / 2
            label = start + 0.114 * np.array([np.cos(middle_angle), np.sin(middle_angle)])
            ax.text(*label, f"$\\theta_{i + 1}$", fontsize=19, ha="center", va="center")

        ax.scatter(*points[:-1].T, s=140, c="white", edgecolors=INK, linewidths=2, zorder=5)
        ax.scatter(end_x, end_y, s=150, c="#dc2626", edgecolors=INK, linewidths=2, zorder=5)
        ax.text(end_x + 0.017, end_y + 0.012, "$(x, y)$", fontsize=17, color="#b91c1c")
        ax.text(-0.021, -0.031, "$O$", fontsize=16, color=INK)

        for start, end, color in [
            ((0, 0), (end_x, 0), COLORS[0]),
            ((end_x, 0), (end_x, end_y), "#dc2626"),
        ]:
            ax.annotate("", xy=end, xytext=start, arrowprops={
                "arrowstyle": "-|>", "color": color, "lw": 2,
            })
        ax.text(end_x / 2, -0.035, "$x$", fontsize=19, color=COLORS[0], ha="center")
        ax.text(end_x + 0.02, end_y / 2, "$y$", fontsize=19, color="#dc2626")

        fig.text(0.70, 0.82, r"$\alpha_i = \sum_{j=1}^{i}\theta_j$", fontsize=21, color=INK)
        for y, formula, color in zip((0.71, 0.62, 0.53), (
            r"$\alpha_1 = \theta_1$",
            r"$\alpha_2 = \theta_1 + \theta_2$",
            r"$\alpha_3 = \theta_1 + \theta_2 + \theta_3$",
        ), COLORS):
            fig.text(0.70, y, formula, fontsize=18, color=color)
        fig.text(0.70, 0.40, r"$\theta_3 < 0$", fontsize=21, color=INK)
        fig.text(0.70, 0.33, r"$\phi = \alpha_3$", fontsize=19, color=INK)

        fig.text(0.075, 0.19,
                 r"$x = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2) + L_3\cos(\theta_1+\theta_2+\theta_3)$",
                 fontsize=19, color=INK)
        fig.text(0.075, 0.10,
                 r"$y = L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2) + L_3\sin(\theta_1+\theta_2+\theta_3)$",
                 fontsize=19, color=INK)
        fig.savefig(OUTPUT, pil_kwargs={"lossless": True})
        plt.close(fig)


if __name__ == "__main__":
    draw_figure()
