"""companion Lab 共用的 GIF 小工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 18) -> ImageFont.ImageFont:
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


def load_font(size: int = 18) -> ImageFont.ImageFont:
    """加载可显示中英文的字体。"""

    return _font(size)


def add_caption(frame: np.ndarray, caption: str) -> np.ndarray:
    """在一帧 RGB 图像上叠加可读的字幕条。"""

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(18)
    margin = 10
    try:
        bbox = draw.textbbox((margin, margin), caption, font=font)
        height = bbox[3] - bbox[1] + 2 * margin
    except AttributeError:
        height = 42
    draw.rectangle((0, 0, image.width, height), fill=(255, 255, 255, 210))
    draw.text((margin, margin), caption, fill=(20, 20, 20, 255), font=font)
    return np.asarray(image)


def fit_scene_frame(
    frame: np.ndarray,
    *,
    output_size: tuple[int, int] | None = None,
    content_target: tuple[float, float] | None = None,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    background_threshold: int = 6,
) -> np.ndarray:
    """把 MuJoCo 黑底画面里的有效内容挪到固定位置。

    `output_size` 使用 PIL 的 `(width, height)` 顺序；`content_target` 是输出图上的
    `(x, y)` 像素坐标。Lab teaser GIF 用它来统一三章里狗的位置。
    """

    source = np.asarray(frame, dtype=np.uint8).copy()
    background = np.all(source < background_threshold, axis=2)
    source[background] = background_rgb

    if output_size is None:
        width, height = source.shape[1], source.shape[0]
    else:
        width, height = output_size
    canvas = np.full((height, width, 3), background_rgb, dtype=np.uint8)

    content = ~background
    if not np.any(content):
        return canvas

    ys, xs = np.nonzero(content)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    if content_target is None:
        target_cx, target_cy = 0.5 * width, 0.5 * height
    else:
        target_cx, target_cy = content_target
    content_cx = 0.5 * (x0 + x1)
    content_cy = 0.5 * (y0 + y1)
    dx = int(round(target_cx - content_cx))
    dy = int(round(target_cy - content_cy))

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(source.shape[1], width - dx)
    src_y1 = min(source.shape[0], height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas

    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _subsample_frames(frames: list[np.ndarray], max_frames: int | None) -> list[np.ndarray]:
    if max_frames is None or len(frames) <= max_frames:
        return frames
    if max_frames <= 0:
        raise ValueError("max_frames 必须为正数")

    indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[int(index)] for index in indices]


def _resize_frame(frame: np.ndarray, width: int | None) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    if width is None or image.width == width:
        return np.asarray(image)
    if width <= 0:
        raise ValueError("width 必须为正数")

    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def _build_global_palette(images: list[Image.Image], palette_colors: int) -> Image.Image:
    """把所有帧拼成一张大图后做 median-cut，得到整段动画共用的全局调色板。

    每帧独立量化时，median-cut 在每张图上挑代表色会跳变，主色是大片蓝、字幕白的小块
    会被不同帧吸成不同的浅蓝 → 整段 GIF 闪烁 + 泛蓝。montage 量化让所有颜色一次到位。
    """

    # 等距抽 32 帧拼成 8 行 × 4 列即可代表整段（再多也只是延长 quantize 时间）。
    sample_count = min(32, len(images))
    if sample_count < len(images):
        idx = np.linspace(0, len(images) - 1, sample_count).round().astype(int)
        sample = [images[int(i)] for i in idx]
    else:
        sample = list(images)

    cols = max(1, int(np.ceil(np.sqrt(sample_count))))
    rows = int(np.ceil(sample_count / cols))
    w, h = sample[0].size
    montage = Image.new("RGB", (cols * w, rows * h))
    for k, image in enumerate(sample):
        montage.paste(image, ((k % cols) * w, (k // cols) * h))
    return montage.quantize(colors=palette_colors, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE)


def write_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int = 15,
    *,
    max_frames: int | None = None,
    width: int | None = None,
    palette_colors: int = 16,
) -> None:
    """把 RGB 帧写成 animated GIF，全局调色板，避免帧间闪烁。"""

    if not frames:
        raise ValueError("write_gif 至少需要一帧图像")
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    if not (2 <= palette_colors <= 256):
        raise ValueError("palette_colors 必须在 2 到 256 之间")

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_images = [Image.fromarray(_resize_frame(f, width)).convert("RGB") for f in _subsample_frames(frames, max_frames)]
    palette_image = _build_global_palette(rgb_images, palette_colors)
    output_frames = [image.quantize(palette=palette_image, dither=Image.Dither.NONE) for image in rgb_images]
    duration_ms = int(round(1000 / fps))
    first, rest = output_frames[0], output_frames[1:]
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
