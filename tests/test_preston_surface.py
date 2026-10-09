"""Physical invariants of the uncalibrated Preston surface integral (no Isaac)."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = _REPO_ROOT
PATH = ROOT / "source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py"
spec = importlib.util.spec_from_file_location("preston_surface", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Surface = module.PlanarPrestonSurface


class PrestonSurfaceTests(unittest.TestCase):
    def surface(self, **kwargs):
        return Surface([[0, 0, 0], [40, 0, 0]], tool_diameter_mm=30, cell_size_mm=1, **kwargs)

    def test_tcp_volume_is_exact_force_times_sliding_speed_times_dt(self):
        for profile in ("uniform", "hertz"):
            surface = self.surface(pressure_profile=profile)
            surface.deposit([10.32, .22, -.1], [3, 4, 999], 20, .008)
            self.assertAlmostEqual(surface.depth.sum() * surface.area, 20 * 5 * .008)

    def test_normal_motion_does_not_become_sliding_removal(self):
        surface = self.surface()
        surface.deposit([10, 0, 0], [0, 0, 5], 20, .008)
        self.assertEqual(surface.depth.sum(), 0)

    def test_tcp_speed_cancels_dwell_on_identical_spatial_samples(self):
        slow, fast = self.surface(), self.surface()
        for x in np.arange(0, 40, .1):
            slow.deposit([x, 0, 0], [3, 0, 0], 20, .1 / 3)
            fast.deposit([x, 0, 0], [9, 0, 0], 20, .1 / 9)
        np.testing.assert_allclose(slow.depth, fast.depth, rtol=1e-12, atol=1e-12)

    def test_spinning_contact_removes_while_tcp_stationary(self):
        surface = self.surface(velocity_model="rotating_disk", spindle_rpm=1000)
        surface.deposit([10, 0, 0], [0, 0, 0], 20, .008)
        first = surface.depth.copy()
        self.assertGreater(first.sum(), 0)
        surface.deposit([10, 0, 0], [0, 0, 0], 20, .008)
        np.testing.assert_allclose(surface.depth, 2 * first)

    def test_zero_rpm_reduces_to_tcp_preston_model(self):
        tcp = self.surface()
        spin = self.surface(velocity_model="rotating_disk", spindle_rpm=0)
        for surface in (tcp, spin):
            surface.deposit([10.2, 0, 0], [3, 4, 0], 20, .008)
        np.testing.assert_array_equal(tcp.depth, spin.depth)

    def test_voltage_or_missing_rpm_cannot_silently_select_a_spin_speed(self):
        with self.assertRaises(ValueError):
            self.surface(velocity_model="rotating_disk")
        with self.assertRaises(ValueError):
            self.surface(spindle_rpm=float("nan"))

    def test_fixed_roi_includes_uncovered_cells_and_reset_clears_depth(self):
        surface = self.surface()
        roi = surface.roi.copy()
        surface.deposit([0, 0, 0], [6, 0, 0], 20, .008)
        metrics = surface.metrics()
        self.assertGreater(metrics["zero_depth_fraction"], .5)
        self.assertGreater(metrics["spatial_depth_cv"], 1)
        self.assertEqual(metrics["spatial_depth_cv"], metrics["equal_cell_volume_cv"])
        surface.reset()
        np.testing.assert_array_equal(surface.roi, roi)
        self.assertEqual(surface.metrics()["zero_depth_fraction"], 1)
        self.assertIsNone(surface.metrics()["spatial_depth_cv"])

    def test_removal_leaving_grid_is_not_redistributed_into_roi(self):
        surface = self.surface()
        surface.deposit([-20, 0, 0], [6, 0, 0], 20, .008)
        self.assertLess(surface.depth.sum() * surface.area, 20 * 6 * .008)
        self.assertAlmostEqual(surface.full_footprint_volume, 20 * 6 * .008)

    def test_nonplanar_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            Surface([[0, 0, 0], [10, 0, 1]])

    def test_control_trace_integral_matches_logged_tcp_removal_and_dropouts(self):
        surface = self.surface()
        count = 101
        xyz = np.c_[np.arange(count) * .048, np.zeros((count, 2))]
        active = np.arange(count) > 0
        force = np.full(count, 20.)
        force[20:30] = 0
        trace = dict(time_s=np.arange(count) * .008, tcp_pose=xyz,
                     measured_speed_mm_s=np.full(count, 6.), normal_force_n=force,
                     polishing_active=active)
        result = surface.integrate_trace(trace)
        self.assertAlmostEqual(result["grid_volume_over_k"], (force[active] * 6 * .008).sum())
        self.assertAlmostEqual(result["processing_time_s"], .8)
        trace["time_s"][10] += .001
        with self.assertRaises(ValueError):
            surface.integrate_trace(trace)


if __name__ == "__main__":
    unittest.main()
