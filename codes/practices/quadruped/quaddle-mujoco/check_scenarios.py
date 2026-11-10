"""Check collision geometry, route clearance and water-force accounting."""
import math
import unittest

import mujoco
import numpy as np

from scenarios import SCENES, ShallowWater, load_scene, make_scene, plan_path, submerged_volume


class SceneChecks(unittest.TestCase):
    def test_surface_matches_real_ray_intersection(self):
        for name in SCENES:
            scene = make_scene(name); model, data, _ = load_scene(scene)
            with self.subTest(scene=name):
                self.assertEqual(model.nu, 4); self.assertEqual(model.neq, 4)
                self.assertAlmostEqual(float(model.body_mass.sum()), .188253, places=6)
                for s in (.0, .21, .33, .45, .67):
                    x = .11
                    origin = np.array([x, -s, .3])
                    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                        np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, np.zeros(1, dtype=np.int32))
                    self.assertGreater(distance, 0.)
                    self.assertAlmostEqual(.3-distance, scene.surface_height(x, -s), places=8)

    def test_downstairs_starts_on_top_platform(self):
        scene = make_scene('stairs_down'); _, data, _ = load_scene(scene)
        self.assertAlmostEqual(data.qpos[2]-scene.surface_height(0, 0), .065)

    def test_known_map_path_clears_inflated_obstacle(self):
        scene = make_scene('avoid'); route = plan_path(scene)
        self.assertEqual(route[0], (0., 0.)); self.assertEqual(route[-1], (0., -scene.goal_m))
        for a, b in zip(route, route[1:]):
            for t in np.linspace(0, 1, 100):
                x, y = (1-t)*np.array(a)+t*np.array(b)
                self.assertFalse(-.13 <= x <= .13 and .34 <= -y <= .62)

    def test_sphere_cap_and_water_forces(self):
        r = .005
        self.assertEqual(submerged_volume(r, .02, .008), 0.)
        self.assertAlmostEqual(submerged_volume(r, 0, .02), 4*math.pi*r**3/3)
        self.assertAlmostEqual(submerged_volume(r, .008, .008), 2*math.pi*r**3/3)
        scene = make_scene('water'); model, data, _ = load_scene(scene)
        data.qpos[1] = -.3; data.qpos[2] = .05; data.qvel[0] = .2
        mujoco.mj_forward(model, data)
        fluid = ShallowWater(model, scene); fluid.apply(data)
        self.assertGreater(fluid.wet_steps, 0)
        self.assertLess(data.qfrc_applied[0], 0.)  # Drag opposes horizontal motion.
        self.assertGreater(data.qfrc_applied[2], 0.)  # Buoyancy opposes gravity.
        first = data.qfrc_applied.copy(); fluid.apply(data)
        np.testing.assert_allclose(data.qfrc_applied, first)  # No force accumulation.

    def test_wet_contact_friction_is_not_overridden_by_robot(self):
        model, data, _ = load_scene(make_scene('water'))
        for _ in range(2000):
            mujoco.mj_step(model, data)
        self.assertGreater(data.ncon, 0)
        for contact in data.contact:
            self.assertAlmostEqual(float(contact.friction[0]), .55)


if __name__ == '__main__':
    unittest.main()
