"""红球 HSV 阈值检测（教程 §8.3 方案 A）。"""

from __future__ import annotations

import cv2
import numpy as np

# 红色在 HSV 横跨 0 度，需要两段并集
_LOW1 = np.array([0, 120, 80], dtype=np.uint8)
_HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
_LOW2 = np.array([170, 120, 80], dtype=np.uint8)
_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

_MIN_AREA = 50


def detect_red_ball(rgb: np.ndarray) -> dict | None:
    """检测图像中的红球，返回 bbox dict 或 None。

    Parameters
    ----------
    rgb : (H, W, 3) uint8 RGB 图像

    Returns
    -------
    dict with keys cx, cy, w, h, area  或  None（未检测到）
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, _LOW1, _HIGH1)
    mask2 = cv2.inRange(hsv, _LOW2, _HIGH2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < _MIN_AREA:
        return None

    x, y, w, h = cv2.boundingRect(best)
    return {
        "cx": x + w / 2,
        "cy": y + h / 2,
        "w": w,
        "h": h,
        "area": area,
    }
