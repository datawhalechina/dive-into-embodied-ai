# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=1.26", "pillow>=10"]
# ///
"""Regenerate Figure 1 with `uv run scripts/generate-cs123-planar-ik.py`.

Both assets use the same geometry. Joint angles are signed rotations from the
preceding link's FORWARD extension, not from a global horizontal or the backward
link. The inset's triangle interior angle is beta = pi - abs(theta_2), for the
principal joint angle in [-pi, pi]. Mathematical labels work in every locale.
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
    / "docs/practices/quadruped/cs123/figs/planar-3dof-ik"
)
LENGTHS = np.array([0.30, 0.25, 0.15])
JOINT_ANGLES = np.deg2rad([52.0, -40.0, 18.0])
COLORS = ("#2563eb", "#15803d", "#d97706")
INK = "#0f172a"
MUTED = "#64748b"


def unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def draw_angle(ax, center, start, end, radius, label, label_radius,
               *, arrow=True, fontsize=19, label_angle=None):
    # Signed sampling also preserves clockwise rotations (theta_2 here).
    angles = np.linspace(start, end, 64)
    arc = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    ax.add_patch(FancyArrowPatch(
        path=MplPath(arc), arrowstyle="-|>" if arrow else "-",
        mutation_scale=12, lw=1.6, color=INK, zorder=4,
    ))
    if label_angle is None:
        label_angle = (start + end) / 2
    ax.text(*(center + label_radius * unit(label_angle)), label,
            fontsize=fontsize, color=INK, ha="center", va="center")


def draw_figure() -> None:
    directions = np.cumsum(JOINT_ANGLES)
    vectors = LENGTHS[:, None] * np.column_stack(
        (np.cos(directions), np.sin(directions))
    )
    points = np.vstack((np.zeros(2), np.cumsum(vectors, axis=0)))
    origin, elbow, wrist, target = points
    phi = directions[-1]

    with plt.rc_context({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "svg.hashsalt": "cs123-planar-ik",
    }):
        fig = plt.figure(figsize=(12, 8.8), dpi=150, facecolor="white")
        fig.text(0.065, 0.94, "3-DoF IK", fontsize=23, weight="bold", color=INK)
        ax = fig.add_axes((0.055, 0.42, 0.63, 0.49))
        ax.set_aspect("equal")
        ax.set_xlim(-0.045, 0.73)
        ax.set_ylim(-0.055, 0.46)
        ax.axis("off")

        for tip, label, offset in [
            ((0.70, 0), "$x$", (0.012, -0.005)),
            ((0, 0.43), "$y$", (-0.025, 0.01)),
        ]:
            ax.annotate("", xy=tip, xytext=origin, arrowprops={
                "arrowstyle": "-|>", "color": INK, "lw": 1.6,
            })
            ax.text(*(np.array(tip) + offset), label, fontsize=17, color=INK)

        ax.plot(*np.vstack((origin, wrist)).T, "--", color=MUTED, lw=1.7)
        r_label = wrist / 2 - 0.035 * unit(np.arctan2(wrist[1], wrist[0]) + np.pi / 2)
        ax.text(*r_label, "$r$", color=MUTED, fontsize=19, ha="center", va="center")
        ax.plot([wrist[0], wrist[0]], [0, wrist[1]], ":", color="#cbd5e1", lw=1.3)
        ax.plot([0, wrist[0]], [wrist[1], wrist[1]], ":", color="#cbd5e1", lw=1.3)

        for i, (start, end, color) in enumerate(zip(points[:-1], points[1:], COLORS)):
            reference = 0.0 if i == 0 else directions[i - 1]
            extension = start + (0.16 if i == 1 else 0.13) * unit(reference)
            ax.plot(*np.vstack((start, extension)).T, "--", color=MUTED, lw=1.6)
            ax.plot(*np.vstack((start, end)).T, color=color, lw=6, solid_capstyle="round")
            normal = unit(directions[i] + np.pi / 2)
            fraction = 0.70 if i == 1 else 0.5
            label = start + fraction * (end - start) + (-0.045 if i == 2 else 0.03) * normal
            ax.text(*label, f"$L_{i + 1}$", color=color, fontsize=20,
                    ha="center", va="center")
            draw_angle(ax, start, reference, directions[i],
                       0.066 if i != 2 else 0.057, f"$\\theta_{i + 1}$",
                       0.112 if i != 2 else 0.097,
                       label_angle=np.deg2rad(14) if i == 0 else None)

        # Subtract the last link: the arrow goes from target toward wrist.
        offset = 0.065 * unit(phi + np.pi / 2)
        ax.annotate("", xy=wrist + offset, xytext=target + offset, arrowprops={
            "arrowstyle": "-|>", "color": COLORS[2], "lw": 2, "linestyle": "--",
        })
        ax.plot(*np.vstack((target, target + np.array([0.14, 0]))).T,
                "--", color=MUTED, lw=1.3)
        ax.plot(*np.vstack((target, target + 0.14 * unit(phi))).T,
                "--", color=COLORS[2], lw=1.3)
        draw_angle(ax, target, 0, phi, 0.085, "$\\phi$", 0.125)

        ax.scatter(*points[:-1].T, s=110, c="white", edgecolors=INK, linewidths=1.8, zorder=5)
        ax.scatter(*target, s=120, c="#dc2626", edgecolors=INK, linewidths=1.8, zorder=5)
        ax.text(-0.022, -0.033, "$O$", fontsize=17, color=INK)
        ax.text(*(wrist + np.array([-0.055, -0.06])), "$W(x', y')$", fontsize=17, color=INK)
        ax.text(*(target + np.array([-0.025, 0.055])), "$T(X, Y, \\phi)$", fontsize=17, color="#b91c1c")

        for y, formula in [
            (0.84, r"$x' = X - L_3\cos\phi$"),
            (0.77, r"$y' = Y - L_3\sin\phi$"),
            (0.68, r"$r^2 = x'^2 + y'^2$"),
            (0.58, r"$\phi = \theta_1 + \theta_2 + \theta_3$"),
            (0.49, r"$\theta_2 < 0$"),
        ]:
            fig.text(0.71, y, formula, fontsize=18, color=INK)

        # The triangle at the elbow uses the BACKWARD L1 ray and forward L2.
        # Its interior angle must not be labelled theta_2.
        inset = fig.add_axes((0.66, 0.055, 0.30, 0.29))
        inset.set_aspect("equal")
        inset.set_xlim(-0.06, 0.49)
        inset.set_ylim(-0.05, 0.35)
        inset.axis("off")
        for start, end, color, label, distance in [
            (origin, elbow, COLORS[0], "$L_1$", 0.035),
            (elbow, wrist, COLORS[1], "$L_2$", 0.035),
            (origin, wrist, MUTED, "$r$", -0.035),
        ]:
            inset.plot(*np.vstack((start, end)).T, color=color, lw=3)
            angle = np.arctan2(*(end - start)[::-1])
            inset.text(*((start + end) / 2 + distance * unit(angle + np.pi / 2)),
                       label, fontsize=18, color=color, ha="center", va="center")
        inset.scatter(*points[:3].T, s=45, c="white", edgecolors=INK, linewidths=1.5, zorder=5)
        backward = origin - elbow
        forward = wrist - elbow
        beta_start = np.arctan2(backward[1], backward[0])
        beta_sweep = np.arctan2(
            backward[0] * forward[1] - backward[1] * forward[0],
            np.dot(backward, forward),
        )
        draw_angle(inset, elbow, beta_start, beta_start + beta_sweep,
                   0.057, "$\\beta$", 0.085, arrow=False,
                   label_angle=directions[1] - np.deg2rad(35))
        inset.text(-0.015, -0.035, "$O$", fontsize=15, color=INK)
        inset.text(*(wrist + np.array([0.025, -0.005])), "$W$", fontsize=15, color=INK)
        fig.text(0.69, 0.37, r"$\beta = \pi - |\theta_2|$", fontsize=20, color=INK)
        fig.text(0.71, 0.33, r"$(-\pi \leq \theta_2 \leq \pi)$", fontsize=13, color=MUTED)

        fig.text(0.075, 0.32, r"$r^2 = L_1^2 + L_2^2 - 2L_1 L_2\cos\beta$", fontsize=20, color=INK)
        fig.text(0.075, 0.20, r"$\cos\theta_2 = \dfrac{r^2 - L_1^2 - L_2^2}{2L_1 L_2}$", fontsize=21, color=INK)
        fig.text(0.075, 0.09, r"$\theta_3 = \phi - \theta_1 - \theta_2$", fontsize=20, color=INK)

        fig.savefig(OUTPUT.with_suffix(".webp"), pil_kwargs={"lossless": True})
        fig.savefig(OUTPUT.with_suffix(".svg"), metadata={
            "Date": None,
            "Title": "3-DoF planar inverse kinematics",
            "Description": (
                "Theta2 is the signed rotation from the forward extension of L1 to L2 "
                "(clockwise and negative in this pose). The triangle interior angle is "
                "beta = pi - abs(theta2), not theta2. Subtract L3 along phi from the "
                "target to obtain the wrist."
            ),
        })
        # Matplotlib's multiline SVG paths include trailing spaces by default.
        svg = OUTPUT.with_suffix(".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
            encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)


if __name__ == "__main__":
    draw_figure()
