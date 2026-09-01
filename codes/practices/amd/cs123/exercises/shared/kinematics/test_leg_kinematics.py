"""shared 4-leg IK helper 的最小数值自检。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[2]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.kinematics.leg_kinematics import HIP_OFFSETS, fk_pupper_leg, ik_pupper_leg  # noqa: E402


def test_four_leg_ik_residual() -> dict[str, float]:
    residuals: dict[str, float] = {}
    for leg, hip in HIP_OFFSETS.items():
        target = hip + np.array((0.015, 0.0, -0.17), dtype=float)
        q = ik_pupper_leg(target, leg=leg)
        err = float(np.linalg.norm(fk_pupper_leg(q, leg=leg) - target))
        assert err < 1e-3, f"{leg} IK residual 过大：{err:.2e} m"
        residuals[leg] = err
    return residuals


def main() -> None:
    residuals = test_four_leg_ik_residual()
    print("shared.kinematics.ik_pupper_leg 检查通过。")
    print({leg: f"{err:.2e} m" for leg, err in residuals.items()})


if __name__ == "__main__":
    main()
