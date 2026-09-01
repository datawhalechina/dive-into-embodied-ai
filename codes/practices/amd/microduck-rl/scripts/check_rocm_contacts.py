#!/usr/bin/env python3
"""Fail fast if the ROCm MuJoCo Warp backend misses dynamic ground contact."""

from __future__ import annotations

import argparse

import mujoco_warp as mjw
import warp as wp
from mujoco_warp import test_data

import mjlab_microduck  # noqa: F401 -- installs the ROCm compatibility hooks

_FALLING_SPHERE_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="ground" type="plane" size="40 40 0.1"/>
    <body name="ball" pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def check(device: str, steps: int = 200) -> tuple[float, int]:
    wp.init()
    selected = wp.get_device(device)
    if not selected.is_hip:
        raise RuntimeError(f"expected a HIP device, got {selected}")
    wp.set_device(selected)

    _mjm, _mjd, model, data = test_data.fixture(xml=_FALLING_SPHERE_XML)
    max_contacts = 0
    for _ in range(steps):
        mjw.step(model, data)
        max_contacts = max(max_contacts, int(data.nacon.numpy()[0]))

    final_z = float(data.qpos.numpy()[0, 2])
    if max_contacts < 1 or not 0.045 <= final_z <= 0.055:
        raise RuntimeError(
            "dynamic contact check failed: "
            f"final_z={final_z:.6f}, max_contacts={max_contacts}"
        )
    return final_z, max_contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    final_z, max_contacts = check(args.device, args.steps)
    print(
        "ROCm dynamic contact check passed: "
        f"final_z={final_z:.6f}, max_contacts={max_contacts}"
    )


if __name__ == "__main__":
    main()
