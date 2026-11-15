"""Check that the collision plane, initial body and measurement axes agree."""
import unittest

import numpy as np

from run_sim import load
from terrain import configure_ground


class TerrainChecks(unittest.TestCase):
    def test_plane_and_initial_pose(self):
        for slope in (-5., 0., 5.):
            with self.subTest(slope=slope):
                model, data = load()
                rotation = configure_ground(model, data, slope)
                # Read actual forward-kinematics output, not just model.geom_quat.
                normal = data.geom('ground').xmat.reshape(3, 3)[:, 2]
                np.testing.assert_allclose(normal, rotation[:, 2], atol=1e-12)
                np.testing.assert_allclose(data.qpos[:3] @ normal, .065, atol=1e-12)
                np.testing.assert_allclose(
                    data.body('body').xmat.reshape(3, 3), rotation, atol=1e-12)
                forward = rotation @ np.array([0., -1., 0.])
                self.assertAlmostEqual(float(forward @ normal), 0.)
                self.assertAlmostEqual(float(forward @ forward), 1.)
                if slope:
                    self.assertGreater(float(forward[2] * slope), 0.)

    def test_invalid_slopes(self):
        model, data = load()
        for slope in (float('nan'), float('inf'), -90., 90.):
            with self.subTest(slope=slope), self.assertRaises(ValueError):
                configure_ground(model, data, slope)


if __name__ == '__main__':
    unittest.main()
