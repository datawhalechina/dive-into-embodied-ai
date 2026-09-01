"""运行 Lab 4 starter，并写出教程观察图。"""

from __future__ import annotations

import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from starter import (  # noqa: E402
    PORTFOLIO_DIR,
    run_pd_sweeps,
    save_heatmap,
    save_stand_z_plot,
    save_sweep_pickle,
    save_zoo_still,
)


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    results = run_pd_sweeps()

    save_heatmap(results, PORTFOLIO_DIR / "pd_heatmap.png")
    save_zoo_still(results, PORTFOLIO_DIR / "pupper_zoo.png")
    save_stand_z_plot(results, PORTFOLIO_DIR / "stand_z_vs_t.png")
    save_sweep_pickle(results, PORTFOLIO_DIR / "pd_stable_gains.pkl")

    print(f"Lab 4 观察图已写入 {PORTFOLIO_DIR}/")
    print("已写出 pd_heatmap.png / pupper_zoo.png / stand_z_vs_t.png")
    for key in ("original", "longleg", "heavy"):
        result = results[key]
        print(
            f"{result.spec.title}: Kp={result.best_gains.kp:g}, Kd={result.best_gains.kd:g}, "
            f"z_std={result.tuned_trace.last_second_z_std * 1000:.2f} mm"
        )


if __name__ == "__main__":
    main()
