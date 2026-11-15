"""Regression for the observed 3-degree uphill startup failure."""
import json
import unittest

from gait_controller import JointController
from run_sim import ROOT
from validate_walk import validate


class StartupChecks(unittest.TestCase):
    def test_uphill_startup_and_walk(self):
        baseline = json.loads((ROOT / 'reference/pre-slope-startup-gait.json').read_text())
        before = validate(seconds=12, slope_deg=3, config=baseline)
        after = validate(seconds=12, slope_deg=3)
        self.assertGreater(before['startup_max_abs_pitch_deg'], 25.)
        self.assertLess(after['startup_max_abs_pitch_deg'], 6.)
        self.assertEqual(after['startup_body_contact_samples'], 0)
        self.assertTrue(after['passed'], after['checks'])

    def test_bad_startup_configuration(self):
        for config in ({'startup_bias_rad': [0.]}, {'startup_ramp_seconds': 0.}):
            from gait_controller import CONFIG
            with self.subTest(config=config), self.assertRaises(ValueError):
                JointController(config={**CONFIG, **config})


if __name__ == '__main__':
    unittest.main()
