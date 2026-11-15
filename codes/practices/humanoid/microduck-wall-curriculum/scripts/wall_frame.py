"""Render a single side-view frame of the wall scene for height comparison.

Usage: python scripts/wall_frame.py OUT.png  (wall height from DIVEWALL_WALL_SINK)
"""
import sys

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def main():
    out = sys.argv[1]
    configure_torch_backends()
    task = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = 1
    v = env_cfg.viewer
    v.camera = "trunk_base"
    v.elevation = 3
    v.azimuth = 95
    v.distance = 1.1
    v.width = 1280
    v.height = 720
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
    env.reset()
    frame = env.render()
    from PIL import Image
    Image.fromarray(frame).save(out)
    env.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
