"""Pupper 变体实验使用的最小控制与绘图工具。"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont


@dataclass(frozen=True)
class PDGains:
    """关节位置控制的比例和微分增益。"""

    kp: float
    kd: float


def load_font(size: int = 18) -> ImageFont.ImageFont:
    """加载支持中英文的字体。"""

    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "Arial Unicode.ttf",
        "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_scene_frame(
    frame: np.ndarray,
    *,
    output_size: tuple[int, int],
    content_target: tuple[float, float],
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    background_threshold: int = 6,
) -> np.ndarray:
    """将 MuJoCo 画面中的有效内容对齐到指定位置。"""

    source = np.asarray(frame, dtype=np.uint8).copy()
    background = np.all(source < background_threshold, axis=2)
    source[background] = background_rgb

    width, height = output_size
    canvas = np.full((height, width, 3), background_rgb, dtype=np.uint8)
    content = ~background
    if not np.any(content):
        return canvas

    ys, xs = np.nonzero(content)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    target_cx, target_cy = content_target
    dx = int(round(target_cx - 0.5 * (x0 + x1)))
    dy = int(round(target_cy - 0.5 * (y0 + y1)))

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(source.shape[1], width - dx)
    src_y1 = min(source.shape[0], height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas

    dst_x0, dst_y0 = src_x0 + dx, src_y0 + dy
    dst_x1, dst_y1 = src_x1 + dx, src_y1 + dy
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return canvas


def apply_theme() -> None:
    """设置紧凑的 Matplotlib 绘图样式。"""

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


def pd_heatmap(
    kp_grid: np.ndarray,
    kd_grid: np.ndarray,
    z_std: np.ndarray,
    *,
    ax,
    title: str,
):
    """绘制以毫米为单位的 PD 稳定性热力图。"""

    values_mm = 1000.0 * np.asarray(z_std, dtype=float)
    image = ax.imshow(values_mm, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(kp_grid)), [f"{kp:g}" for kp in kp_grid])
    ax.set_yticks(np.arange(len(kd_grid)), [f"{kd:g}" for kd in kd_grid])
    ax.set_xlabel("$K_p$")
    ax.set_ylabel("$K_d$")
    ax.set_title(title)
    for row in range(values_mm.shape[0]):
        for column in range(values_mm.shape[1]):
            ax.text(
                column,
                row,
                f"{values_mm[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    return image
