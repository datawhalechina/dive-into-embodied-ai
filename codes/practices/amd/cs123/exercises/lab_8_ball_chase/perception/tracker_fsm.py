"""P 控制器 visual_servo + TrackerFSM（教程 §8.4–§8.5）。"""

from __future__ import annotations

import enum

import numpy as np

KP_YAW = 1.0
KP_VX = 0.4
H_REF = 30  # 像素：球在画面中占这个高度时认为"够近了"
SEARCH_WZ = 0.6  # SEARCHING 状态下的原地转速 rad/s
LOST_TIMEOUT = 2.0  # 丢失目标超时 → STOPPED


class FSMState(enum.Enum):
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    STOPPED = "STOPPED"


def visual_servo(
    box: dict,
    image_w: int,
    image_h: int,
) -> tuple[float, float]:
    """根据 bbox 计算 (vx, wz) 命令。

    Parameters
    ----------
    box : detect_red_ball 返回的 dict（cx, cy, w, h, area）
    image_w, image_h : 图像尺寸

    Returns
    -------
    (vx, wz)
    """
    e_yaw = (box["cx"] - image_w / 2) / (image_w / 2)
    wz = -KP_YAW * e_yaw

    e_size = max(0.0, 1.0 - box["h"] / H_REF)
    vx = KP_VX * e_size

    return float(vx), float(wz)


class TrackerFSM:
    """三态追踪状态机：SEARCHING → TRACKING → STOPPED。"""

    def __init__(self) -> None:
        self.state = FSMState.SEARCHING
        self._last_seen: float = 0.0
        self._last_box: dict | None = None

    def reset(self) -> None:
        self.state = FSMState.SEARCHING
        self._last_seen = 0.0
        self._last_box = None

    def step(
        self,
        t: float,
        boxes: list[dict],
        image_w: int,
        image_h: int,
    ) -> tuple[FSMState, float, float]:
        """根据当前检测结果更新状态，返回 (state, vx, wz)。

        Parameters
        ----------
        t : 当前仿真时间
        boxes : detect_red_ball 返回的 bbox 列表（0 或 1 个元素）
        image_w, image_h : 图像尺寸
        """
        box = self._select(boxes)

        if box is not None:
            self._last_seen = t
            self._last_box = box
            self.state = FSMState.TRACKING
            vx, wz = visual_servo(box, image_w, image_h)
            return self.state, vx, wz

        lost_duration = t - self._last_seen

        if self.state == FSMState.TRACKING:
            if lost_duration > LOST_TIMEOUT:
                self.state = FSMState.SEARCHING
            else:
                # 短暂丢失：保持上一帧的命令方向，减速
                if self._last_box is not None:
                    vx, wz = visual_servo(self._last_box, image_w, image_h)
                    return self.state, vx * 0.5, wz * 0.5
                return self.state, 0.0, 0.0

        if self.state == FSMState.SEARCHING:
            if lost_duration > LOST_TIMEOUT * 3:
                self.state = FSMState.STOPPED
                return self.state, 0.0, 0.0
            return self.state, 0.0, SEARCH_WZ

        # STOPPED
        return self.state, 0.0, 0.0

    def _select(self, boxes: list[dict]) -> dict | None:
        """多目标选择：和上一帧 bbox 中心欧氏距离最近。"""
        if not boxes:
            return None
        if len(boxes) == 1:
            return boxes[0]
        if self._last_box is None:
            return max(boxes, key=lambda b: b["area"])
        lx, ly = self._last_box["cx"], self._last_box["cy"]
        return min(boxes, key=lambda b: (b["cx"] - lx) ** 2 + (b["cy"] - ly) ** 2)
