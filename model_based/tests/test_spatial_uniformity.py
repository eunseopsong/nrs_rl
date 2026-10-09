"""Physical invariants and control limits of the new experimental policy."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import (SignedSpeedLimiter, LocalReturn, SpatialUniformityActor,
    PlanarPrestonSurface, DoseGeometry, action_numpy, assess, ROOT)


class SpatialUniformityTests(unittest.TestCase):
    def test_no_rotation_static_dwell_cannot_remove_material(self):
        surface = PlanarPrestonSurface([[0., 0., 0.], [30., 0., 0.]])
        for _ in range(100):
            surface.deposit([10., 0., 0.], [0., 0., 0.], 26., .008)
        self.assertEqual(surface.full_footprint_volume, 0.)
        self.assertEqual(float(surface.depth.sum()), 0.)

    def test_retiming_preserves_depth_but_force_and_extra_sliding_change_it(self):
        maps = []
        for factor, force, passes in [(1., 20., 1), (2., 20., 1), (1., 24., 1), (1., 20., 2)]:
            surface = PlanarPrestonSurface([[0., 0., 0.], [30., 0., 0.]])
            for _ in range(passes):
                for x in np.arange(.05, 30., .1):
                    surface.deposit([x, 0., 0.], [10./factor, 0., 0.], force, .01*factor)
            maps.append(surface.depth.copy())
        np.testing.assert_allclose(maps[0], maps[1], atol=1e-12)
        np.testing.assert_allclose(maps[2], maps[0]*1.2, atol=1e-12)
        np.testing.assert_allclose(maps[3], maps[0]*2., atol=1e-12)

    def test_signed_feed_reversal_limits_reset_and_native_forward_parity(self):
        path = next((ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/y2_control_pybind/y2_control_py').glob('_y2_control_pybind*.so'))
        spec = importlib.util.spec_from_file_location('_y2_control_pybind', path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        native = module.PpoSpeedLimiter(.008, 9., 16., 160.)
        limiter = SignedSpeedLimiter()
        for request in np.r_[np.full(300, 6.), np.full(300, 3.), np.full(300, 9.)]:
            self.assertAlmostEqual(limiter.step(float(request)), native.step(float(request)), places=12)
        speeds = [limiter.step(float(v)) for v in np.r_[np.full(500, -3.), np.full(500, 9.)]]
        acceleration = np.diff(speeds)/.008
        jerk = np.diff(acceleration)/.008
        self.assertLessEqual(np.max(np.abs(acceleration)), 16.+1e-8)
        self.assertLessEqual(np.max(np.abs(jerk)), 160.+1e-6)
        self.assertLess(min(speeds), -2.9)
        self.assertEqual(limiter.step(9., False), 0.)
        self.assertEqual(limiter.acceleration, 0.)

    def test_local_return_is_bounded_and_not_a_second_pass(self):
        route, limiter = LocalReturn(697.5), SignedSpeedLimiter()
        cursor = 0.;minimum_lag = 0.;maximum_lag = 0.
        for _ in range(21000):
            requested = route.request(cursor, 6., 1.)
            cursor = route.advance(cursor, limiter.step(requested)*.008)
            maximum_lag = max(maximum_lag, route.frontier-cursor)
            if cursor >= route.length:
                break
        self.assertEqual(cursor, route.length)
        self.assertGreater(route.events, 0)
        self.assertLessEqual(route.reverse_mm, .05*route.length)
        self.assertLess(maximum_lag, 10.)
        route.reset()
        self.assertEqual((route.frontier, route.reverse_mm, route.events), (0., 0., 0))

    def test_export_parity_and_contact_return_gate(self):
        p = [1.2, .7, -.06, .9, .02, .01, .1, 2., .025]
        obs = np.random.default_rng(42).normal(size=(1024, 16)).astype(np.float32)
        actor = torch.jit.script(SpatialUniformityActor(p))
        np.testing.assert_allclose(actor(torch.tensor(obs)).numpy(), action_numpy(obs, p), atol=1e-6)
        obs[:, 10] = 0.
        self.assertTrue(np.all(actor(torch.tensor(obs)).numpy()[:, 2] < 0.))

    def test_low_volume_and_unfinished_candidates_cannot_win(self):
        base = dict(spatial_cv=.4, roi_volume_over_k=100., time_s=100., force_mean_n=20.,
                    force_peak_n=26., distance_mm=700., zero_fraction=0., rate_cv=.2, completed=True)
        self.assertFalse(assess({**base, 'spatial_cv': .1, 'roi_volume_over_k': 50.}, base)['passed'])
        self.assertFalse(assess({**base, 'completed': False}, base)['passed'])


if __name__ == '__main__':
    unittest.main()
