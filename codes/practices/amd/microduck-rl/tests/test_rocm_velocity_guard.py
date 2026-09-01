"""AMD-specific safety guard configuration."""

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_velocity_task_terminates_below_ground():
    cfg = make_microduck_velocity_env_cfg()
    term = cfg.terminations["below_ground"]
    assert term.params == {"min_height": -0.05}
    assert term.time_out is False
