"""Lab 4 数值检查：验证 starter.py 的变体、站姿和 PD 扫描。"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.controllers.pd_controller import PDGains  # noqa: E402
from starter import (  # noqa: E402
    KD_GRID,
    KP_GRID,
    SKELETON_PATH,
    VARIANTS,
    _variant_spec,
    base_height_for_pose,
    find_stand_pose,
    load_model,
    make_all_variants,
    make_variant,
    find_stable_pd_gains,
    simulate_stand,
)


def _geom_length(model: mujoco.MjModel, name: str) -> float:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise AssertionError(f"缺少 geom {name}")
    return float(2.0 * model.geom_size[geom_id, 1])


def test_variant_compiles_and_scales() -> float:
    original_path = make_variant("original", leg_scale=1.0)
    long_path = make_variant("longleg", leg_scale=1.5)
    original = mujoco.MjModel.from_xml_path(str(original_path))
    longleg = mujoco.MjModel.from_xml_path(str(long_path))
    ratio = _geom_length(longleg, "FL_thigh_capsule") / _geom_length(original, "FL_thigh_capsule")
    assert abs(ratio - 1.5) < 0.03, f"long-leg thigh fromto 比例不对：ratio={ratio:.3f}"
    return ratio


def test_three_variants_stand_under_best_pd() -> dict[str, float]:
    paths = make_all_variants()
    z_std_mm: dict[str, float] = {}
    for key, spec in VARIANTS.items():
        model, _ = load_model(paths[key])
        stand_pose = find_stand_pose(model, spec.leg_scale)
        kp, kd, _grid = find_stable_pd_gains(model, stand_pose, KP_GRID, KD_GRID)
        trace = simulate_stand(
            model,
            stand_pose,
            PDGains(kp=kp, kd=kd),
            seconds=6.0,
            base_height=base_height_for_pose(model, stand_pose),
            disturbance=True,
        )
        z_std = trace.last_second_z_std
        z_std_mm[key] = z_std * 1000.0
        assert z_std < 0.005, f"{spec.title} 最后 1 秒 base z 抖动过大：{z_std * 1000:.2f} mm"
    return z_std_mm


def test_pd_heatmap_selects_reasonable_original() -> tuple[float, float, float, float]:
    path = make_variant("original", leg_scale=1.0)
    model, _ = load_model(path)
    stand_pose = find_stand_pose(model, 1.0)
    kp_grid = np.array((5.0, 10.0, 30.0, 60.0, 120.0), dtype=float)
    kd_grid = np.array((0.5, 1.0, 2.0, 5.0), dtype=float)
    kp, kd, grid = find_stable_pd_gains(model, stand_pose, kp_grid, kd_grid)
    soft = float(grid[np.where(kd_grid == 1.0)[0][0], np.where(kp_grid == 5.0)[0][0]])
    best = float(grid[np.where(kd_grid == kd)[0][0], np.where(kp_grid == kp)[0][0]])
    assert kp >= 30.0, f"原版 Pupper 的较优 Kp 不应过软：Kp={kp:g}"
    assert kd >= 1.0, f"原版 Pupper 的较优 Kd 不应过低：Kd={kd:g}"
    assert soft > best + 2e-4, f"过软格没有显著差于较优参数：soft={soft:.4g}, best={best:.4g}"
    return kp, kd, soft, best


def test_skeleton_include_and_mass_consistency() -> tuple[int, float]:
    paths = make_all_variants()
    source = SKELETON_PATH.read_text(encoding="utf-8")
    assert "<freejoint" in source and source.count("_HAA") >= 4, "skeleton.xml 必须是完整 12-DoF Pupper"
    for spec in VARIANTS.values():
        variant_source = paths[spec.key].read_text(encoding="utf-8")
        assert '<include file="../../shared/models/skeleton.xml"/>' in variant_source
        assert len(variant_source.strip().splitlines()) <= 30, f"{spec.file_name} 超过 30 行"

    original = mujoco.MjModel.from_xml_path(str(paths["original"]))
    longleg = mujoco.MjModel.from_xml_path(str(paths["longleg"]))
    heavy = mujoco.MjModel.from_xml_path(str(paths["heavy"]))
    assert original.nq == longleg.nq == heavy.nq, "三份变体结构 nq 必须一致"
    mass_ratio = float(heavy.body_mass.sum() / original.body_mass.sum())
    assert 1.30 < mass_ratio < 1.80, f"heavy 总质量倍率不合理：{mass_ratio:.2f}"
    return original.nq, mass_ratio


def main() -> None:
    scale_ratio = test_variant_compiles_and_scales()
    z_std_mm = test_three_variants_stand_under_best_pd()
    kp, kd, soft, best = test_pd_heatmap_selects_reasonable_original()
    nq, mass_ratio = test_skeleton_include_and_mass_consistency()
    print("Lab 4 检查全部通过。")
    print(f"long-leg thigh ratio = {scale_ratio:.3f}")
    print(f"best original = (Kp={kp:g}, Kd={kd:g}), soft={soft * 1000:.2f} mm, best={best * 1000:.2f} mm")
    print(f"z_std mm = {z_std_mm}")
    print(f"nq = {nq}, heavy mass ratio = {mass_ratio:.2f}")


if __name__ == "__main__":
    main()
