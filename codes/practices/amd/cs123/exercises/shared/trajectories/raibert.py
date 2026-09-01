"""Raibert 踏步会用到的相位 helpers。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseState:
    """一个周期内的 stance / swing 相位。"""

    phase: float
    in_stance: bool
    local_phase: float


def stance_swing_phase(t: float, period: float, duty: float = 0.5) -> PhaseState:
    """把时间映射成 stance / swing 相位。

    `phase` 是整周期内的 0..1，`local_phase` 是当前半段内的 0..1。
    """

    if period <= 0.0:
        raise ValueError("period 必须为正数")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty 必须在 (0, 1) 内")

    phase = (float(t) % period) / period
    if phase < duty:
        return PhaseState(
            phase=phase,
            in_stance=True,
            local_phase=phase / duty,
        )
    return PhaseState(
        phase=phase,
        in_stance=False,
        local_phase=(phase - duty) / (1.0 - duty),
    )
