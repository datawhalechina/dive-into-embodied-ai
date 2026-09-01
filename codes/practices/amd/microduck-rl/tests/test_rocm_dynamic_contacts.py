"""GPU regression test for ROCm MuJoCo Warp dynamic broadphase."""

import pytest
import warp as wp


@pytest.mark.skipif("+rocm" not in wp.__version__, reason="not the ROCm Warp fork")
def test_falling_sphere_reaches_ground_on_hip():
    from scripts.check_rocm_contacts import check

    final_z, max_contacts = check("cuda:0")
    assert max_contacts >= 1
    assert final_z == pytest.approx(0.05, abs=0.005)
