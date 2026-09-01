"""填完 starter 三处空白后运行的 Lab 2 数值检查。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.kinematics import Transform
from starter import (
    T_BASE_HIP_FIXED,
    T_HIP_UPPER_FIXED,
    T_LOWER_FOOT,
    T_UPPER_LOWER_FIXED,
    T_WORLD_BASE,
    fk_validate,
    sample_workspace,
)


def _broken_hfe_axis_fk(theta) -> np.ndarray:
    """故意把 HFE 那段写成绕 x，测试校验是否真的能抓错。"""

    theta = np.asarray(theta, dtype=float)
    haa, hfe, kfe = theta
    T_world_HAA = T_WORLD_BASE @ T_BASE_HIP_FIXED @ Transform.rot_z(haa)
    T_HAA_HFE = T_HIP_UPPER_FIXED @ Transform.rot_x(hfe)
    T_HFE_KFE = T_UPPER_LOWER_FIXED @ Transform.rot_z(kfe)
    T_world_foot = T_world_HAA @ T_HAA_HFE @ T_HFE_KFE @ T_LOWER_FOOT
    return T_world_foot[:3, 3].copy()


def test_task_c_matches_mujoco() -> None:
    max_err = fk_validate(seed=0, n=100)
    assert max_err < 1e-10, f"任务 C 误差太大：max_err={max_err:.2e} m"


def test_task_c_wrong_axis_fails() -> None:
    max_err = fk_validate(seed=0, n=100, fk_fn=_broken_hfe_axis_fk)
    assert max_err > 1e-3, f"证伪测试没有拉开误差：max_err={max_err:.2e} m"


def test_task_e_workspace_z_bounds() -> None:
    points = sample_workspace(20_000, seed=0)
    assert points.shape == (20_000, 3), f"工作空间点云形状不对：{points.shape}"
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    assert z_min < 0.35, f"足端最低点不合理：z_min={z_min:.3f} m"
    assert z_max > 0.65, f"足端最高点不合理：z_max={z_max:.3f} m"


def main() -> None:
    test_task_c_matches_mujoco()
    test_task_c_wrong_axis_fails()
    test_task_e_workspace_z_bounds()
    print("Lab 2 检查全部通过。")


if __name__ == "__main__":
    main()
