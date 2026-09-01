"""填完 starter 三处空白后运行的 Lab 3 数值检查。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.kinematics import fk_leg  # noqa: E402
from starter import (  # noqa: E402
    PERIOD,
    check_scene_mounted,
    dls_ik,
    load_model,
    raibert_foot_traj,
    simulate_stepping,
)


def test_dls_reaches_fk_target() -> float:
    q_truth = np.array((0.0, -0.92, 1.34), dtype=float)
    target = fk_leg(q_truth)
    result = dls_ik(np.array((0.0, -0.65, 1.05), dtype=float), target, tol=1e-3, max_iter=50)
    err = float(np.linalg.norm(fk_leg(result.q) - target))
    assert err < 1e-3, f"DLS 没收敛到 FK 目标：err={err:.2e} m"
    return err


def test_newton_without_damping_fails_near_singularity() -> float:
    q0 = np.array((0.0, 0.0, 0.0), dtype=float)
    target = fk_leg(q0) + np.array((0.0, 0.0, -0.10), dtype=float)
    result = dls_ik(q0, target, lam=0.0, step=1.0, tol=1e-8, max_iter=8)
    assert result.iters == 8, f"lam=0 应该跑到 max_iter，实际 iters={result.iters}"
    assert result.residual > 1e-2, f"lam=0 证伪残差太小：residual={result.residual:.2e} m"
    return result.residual


def test_raibert_periodic_closure() -> float:
    err = float(np.linalg.norm(raibert_foot_traj(0.0) - raibert_foot_traj(PERIOD)))
    assert err < 1e-9, f"Raibert 轨迹周期不闭合：err={err:.2e} m"
    return err


def test_closed_loop_weld_keeps_base_fixed() -> float:
    model, _ = load_model()
    check_scene_mounted(model)
    trace = simulate_stepping(seconds=8.0, capture_frames=False)
    z_std = float(np.std(trace.base_z))
    assert z_std < 1e-6, f"base z 抖动过大，weld 可能没锁住：std={z_std:.2e} m"
    return z_std


def main() -> None:
    ik_err = test_dls_reaches_fk_target()
    newton_residual = test_newton_without_damping_fails_near_singularity()
    closure_err = test_raibert_periodic_closure()
    base_z_std = test_closed_loop_weld_keeps_base_fixed()
    print("Lab 3 检查全部通过。")
    print(f"DLS FK target err = {ik_err:.2e} m")
    print(f"lam=0 证伪 residual = {newton_residual:.2e} m")
    print(f"Raibert closure err = {closure_err:.2e} m")
    print(f"base z std = {base_z_std:.2e} m")


if __name__ == "__main__":
    main()
