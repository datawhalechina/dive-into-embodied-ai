"""配套 Lab 共用的 Matplotlib 工具函数。"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from shared.controllers.pd_controller import BodePoint


def apply_theme() -> None:
    """给 Lab 图表套一层紧凑、易读的样式。"""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "font.sans-serif": [
                "PingFang SC",
                "Heiti TC",
                "Songti SC",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "savefig.dpi": 150,
        }
    )


def bode_figure(
    points: Iterable[BodePoint],
    *,
    title: str = "Pupper HFE 正弦跟踪 Bode 图",
    subtitle: str | None = None,
):
    """根据正弦跟踪测得的频点画增益和相位。"""

    apply_theme()
    pts = sorted(points, key=lambda p: p.frequency_hz)
    freqs = np.array([p.frequency_hz for p in pts], dtype=float)
    gain_db = np.array([p.gain_db for p in pts], dtype=float)
    phase = np.array([p.phase_deg for p in pts], dtype=float)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    ax_gain.semilogx(freqs, gain_db, "o-", color="tab:blue", label="增益")
    ax_gain.axhline(-3.0, color="tab:red", linestyle="--", linewidth=1.2, label="-3 dB")
    ax_gain.set_ylabel("增益 (dB)")
    ax_gain.legend(loc="best")

    crossing = _bandwidth_crossing(freqs, gain_db)
    if crossing is not None:
        ax_gain.axvline(crossing, color="tab:red", linestyle=":", linewidth=1.0)
        ax_gain.annotate(
            f"带宽约 {crossing:.2g} Hz",
            xy=(crossing, -3.0),
            xytext=(8, -24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
            color="tab:red",
        )

    ax_phase.semilogx(freqs, phase, "o-", color="tab:orange", label="相位差")
    ax_phase.axhline(0.0, color="0.3", linewidth=0.8)
    ax_phase.set_xlabel("频率 (Hz)")
    ax_phase.set_ylabel("相位 (deg)")
    ax_phase.legend(loc="best")
    ax_phase.set_xticks(freqs)
    ax_phase.set_xticklabels([f"{f:g}" for f in freqs])

    fig.suptitle(title)
    if subtitle:
        ax_gain.set_title(subtitle, fontsize=10, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def workspace_scatter(
    fig,
    points: np.ndarray,
    *,
    title: str,
    point_size: float = 1.0,
    alpha: float = 0.1,
):
    """在已有 figure 上画 3D 工作空间散点，返回 axes。"""

    apply_theme()
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 必须是形状为 (n, 3) 的数组")

    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=pts[:, 2],
        s=point_size,
        alpha=alpha,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-55)
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="z [m]")

    ranges = np.ptp(pts, axis=0)
    center = np.mean(pts, axis=0)
    radius = max(float(np.max(ranges)) * 0.5, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    return ax


def pd_heatmap(fig, kp_grid: np.ndarray, kd_grid: np.ndarray, z_std: np.ndarray, *, ax=None, title: str = ""):
    """在已有 figure 上画 PD 甜点 heatmap，颜色单位是 mm。"""

    apply_theme()
    if ax is None:
        ax = fig.add_subplot(111)
    values_mm = 1000.0 * np.asarray(z_std, dtype=float)
    im = ax.imshow(values_mm, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(kp_grid)), [f"{kp:g}" for kp in kp_grid])
    ax.set_yticks(np.arange(len(kd_grid)), [f"{kd:g}" for kd in kd_grid])
    ax.set_xlabel("$K_p$")
    ax.set_ylabel("$K_d$")
    ax.set_title(title)
    for i in range(values_mm.shape[0]):
        for j in range(values_mm.shape[1]):
            ax.text(j, i, f"{values_mm[i, j]:.1f}", ha="center", va="center", fontsize=8)
    return im


def gait_gantt(figure, gait_dict: dict, *, T_window: float = 2.0):
    """画单个 gait 的 4-leg stance/swing 甘特图，深色块表示 stance。"""

    apply_theme()
    ax = figure.add_subplot(111)
    legs = ("FL", "FR", "RL", "RR")
    offsets = gait_dict["offsets"]
    duty = float(gait_dict["duty"])
    T_cycle = float(gait_dict["T_cycle"])
    for row, leg in enumerate(legs):
        t = -float(offsets[leg]) * T_cycle
        while t < T_window:
            start = max(t, 0.0)
            end = min(t + duty * T_cycle, T_window)
            if end > start:
                ax.broken_barh([(start, end - start)], (row - 0.36, 0.72), facecolors="0.15")
            t += T_cycle
    ax.set_xlim(0.0, T_window)
    ax.set_ylim(-0.6, len(legs) - 0.4)
    ax.set_yticks(range(len(legs)), legs)
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("腿")
    ax.set_title(gait_dict.get("name", "gait"))
    ax.grid(True, axis="x", alpha=0.28)
    ax.grid(False, axis="y")
    return ax


def _bandwidth_crossing(freqs: np.ndarray, gain_db: np.ndarray) -> float | None:
    below = np.where(gain_db <= -3.0)[0]
    if len(below) == 0:
        return None
    idx = int(below[0])
    if idx == 0:
        return float(freqs[0])
    x0, x1 = np.log10(freqs[idx - 1]), np.log10(freqs[idx])
    y0, y1 = gain_db[idx - 1], gain_db[idx]
    if abs(y1 - y0) < 1e-12:
        return float(freqs[idx])
    x = x0 + (-3.0 - y0) * (x1 - x0) / (y1 - y0)
    return float(10**x)
