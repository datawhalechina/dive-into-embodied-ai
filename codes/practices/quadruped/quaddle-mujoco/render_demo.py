"""Record actual stepped dynamics using MuJoCo's offscreen renderer."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_sim import ROOT, load, camera_and_options
from gait_controller import JointController, CONFIG
from terrain import configure_ground


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--mode", choices=["motion", "walk", "stand"], default="walk")
    parser.add_argument("--config", type=Path, help="Optional gait JSON for comparison")
    parser.add_argument("--output", default=None, help="Output basename, without extension")
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--slope-deg", type=float, default=0.)
    parser.add_argument("--friction", type=float, help="Override sliding friction on every geometry")
    args = parser.parse_args()
    m, d = load()
    ground_rotation = configure_ground(m, d, args.slope_deg)
    forward_axis = ground_rotation @ np.array([0., -1., 0.])
    if args.friction is not None:
        if not np.isfinite(args.friction) or args.friction <= 0:
            parser.error("--friction must be finite and positive")
        m.geom_friction[:, 0] = args.friction
    mujoco.mj_forward(m, d)
    cam, opt = camera_and_options(d)
    config = json.loads(args.config.read_text()) if args.config else CONFIG
    controller = JointController(args.mode, config=config)
    cam.azimuth = args.azimuth
    video_name = args.output or ("quaddle-walking" if args.mode == "walk" else "quaddle-mujoco")
    origin = None
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    font_path = next((p for p in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ) if p.exists()), None)
    font = ImageFont.truetype(str(font_path), 21) if font_path else ImageFont.load_default(size=21)
    small = ImageFont.truetype(str(font_path), 15) if font_path else ImageFont.load_default(size=15)
    # 25 fps divides the 1 kHz simulation timestep exactly.
    with mujoco.Renderer(m, height=720, width=960) as renderer, imageio.get_writer(
        out / f"{video_name}.mp4", fps=25, codec="libx264", quality=8, macro_block_size=16
    ) as writer:
        for i in range(round(args.seconds * 25)):
            for _ in range(40):
                controller(d)
                mujoco.mj_step(m, d)
                if origin is None and d.time >= config["settle_seconds"]:
                    origin = d.qpos[:3].copy()
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.body("body").xpos
            renderer.update_scene(d, camera=cam, scene_option=opt)
            # Fixed world markers make progress visible even with a tracking camera.
            if args.mode == "walk":
                scene = renderer.scene
                for j in range(26):
                    for x in (-.11, .11):
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([.006, .002, .0005]), ground_rotation @ np.array([x, .026 - j*.1, .001]),
                            ground_rotation.flatten(), np.array([.2, .8, .9, 1.]))
                        scene.ngeom += 1
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 89), fill=(14, 21, 33))
            draw.text((24, 15), "QUADDLE  /  MuJoCo", font=font, fill=(244, 247, 250))
            label = {"coordinated_trot": "Alternating diagonal gait",
                     "coordinated_crawl": "Four-phase alternating gait"}.get(config.get("name"), "Walking + heading feedback")
            subtitle = f"{label} | {config['frequency_hz']:.2f} Hz | Ground markers: 10 cm" if args.mode == "walk" else "Official CAD + local dynamics adapter | 4 actuators | 4 loop constraints"
            draw.text((24, 50), subtitle, font=small, fill=(170, 193, 211))
            draw.rectangle((0, 668, 960, 720), fill=(14, 21, 33))
            progress = (d.qpos[:3] - origin) @ forward_axis if origin is not None else 0.
            caption = f"t={d.time:05.2f}s | Forward={progress:.2f}m | Slope={args.slope_deg:+.1f}deg | Uncalibrated simulation"
            draw.text((24, 678), caption, font=small, fill=(227, 190, 120))
            writer.append_data(np.asarray(frame))
            if i == 100:
                frame.save(out / (f"{args.output}-preview.webp" if args.output else ("walking-preview.webp" if args.mode == "walk" else "preview.webp")), quality=85)
            if i % 50 == 0:
                print(f"Rendered {i + 1}/{round(args.seconds * 25)}", flush=True)
    print(out / f"{video_name}.mp4", flush=True)


if __name__ == "__main__":
    main()
