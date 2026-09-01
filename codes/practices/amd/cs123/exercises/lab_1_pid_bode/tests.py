"""填完 starter 三处 TODO 后运行的 Lab 1 数值检查。"""

from __future__ import annotations

from starter import AMPLITUDE_RAD, default_gains, run_bode_sweep, run_constant_hold


def test_task_a_constant_hold() -> None:
    gains, inertia = default_gains()
    trace = run_constant_hold(gains=gains, inertia=inertia)
    tail_error = abs(float(trace.q[-1] - AMPLITUDE_RAD))
    assert tail_error < 0.02, f"任务 A 没有稳住：最终误差={tail_error:.4f} rad"


def test_task_d_low_frequency_transmits() -> None:
    gains, inertia = default_gains()
    points = run_bode_sweep(gains=gains, inertia=inertia, frequencies_hz=(0.2,))
    assert points[0].gain > 0.95, f"0.2 Hz 增益太低：{points[0].gain:.3f}"


def test_task_d_high_frequency_rolls_off() -> None:
    gains, inertia = default_gains()
    points = run_bode_sweep(gains=gains, inertia=inertia, frequencies_hz=(8.0,))
    assert points[0].gain < 0.5, f"8 Hz 增益太高：{points[0].gain:.3f}"


def main() -> None:
    test_task_a_constant_hold()
    test_task_d_low_frequency_transmits()
    test_task_d_high_frequency_rolls_off()
    print("Lab 1 检查全部通过。")


if __name__ == "__main__":
    main()
