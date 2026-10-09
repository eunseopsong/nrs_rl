"""Physical invariants for the explicitly normalized rotation-dominated limit."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import unittest
import numpy as np
from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface, preston


class RotationDepthTests(unittest.TestCase):
    path = [[0, 0, 0], [40, 0, 0]]

    def test_same_path_faster_feed_reduces_depth_inversely(self):
        slow, fast = (RotationDominatedSurface(self.path) for _ in range(2))
        for x in np.arange(0, 40, .2):
            slow.deposit([x, 0, 0], [3, 0, 0], 20, .2/3)
            fast.deposit([x, 0, 0], [9, 0, 0], 20, .2/9)
        np.testing.assert_allclose(slow.depth, 3*fast.depth, atol=1e-12)

    def test_rpm_factor_cancels_exactly_in_stationary_contact(self):
        normalized = RotationDominatedSurface(self.path)
        normalized.deposit([10.2, .3, 0], [0, 0, 0], 20, .008)
        for rpm in (100, 1000, 10000):
            full = preston.PlanarPrestonSurface(self.path, velocity_model='rotating_disk', spindle_rpm=rpm)
            full.deposit([10.2, .3, 0], [0, 0, 0], 20, .008)
            np.testing.assert_allclose(full.depth / (rpm*2*np.pi/60), normalized.depth, atol=1e-12)
        self.assertGreater(normalized.depth.sum(), 0)
        self.assertIsNone(normalized.metadata['spindle_rpm'])
        self.assertIn('mean_depth_over_k_omega', normalized.metrics())
        self.assertNotIn('mean_depth_over_k', normalized.metrics())

    def test_depth_compensation_force_direction_differs_from_rate_compensation(self):
        reference, inverse_force, proportional_force = (RotationDominatedSurface(self.path) for _ in range(3))
        for x in np.arange(0, 40, .2):
            reference.deposit([x, 0, 0], [6, 0, 0], 20, .2/6)
            inverse_force.deposit([x, 0, 0], [3, 0, 0], 40, .2/3)
            proportional_force.deposit([x, 0, 0], [12, 0, 0], 40, .2/12)
        np.testing.assert_allclose(proportional_force.depth, reference.depth, atol=1e-12)
        np.testing.assert_allclose(inverse_force.depth, 4*reference.depth, atol=1e-12)

    def test_fault_or_no_contact_cannot_deposit(self):
        surface = RotationDominatedSurface(self.path)
        surface.deposit([0, 0, 0], [0, 0, 0], 20, .008, contact=False)
        self.assertEqual(surface.depth.sum(), 0)
        with self.assertRaises(ValueError):
            surface.deposit([0, 0, 0], [np.nan, 0, 0], 20, .008)
        with self.assertRaises(ValueError):
            RotationDominatedSurface(self.path, spindle_rpm=30)


if __name__ == '__main__':
    unittest.main()
