"""Rate preservation, causal spindle response, and independent neural outputs."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.rpm_control import RPMConfig, VirtualSpindle, RateObjective
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.feed_rpm_ppo import FeedRPMActorCritic, export_actor
from nrs_rl.tasks.manager_based.nrs_rl.utils.preston_surface import PlanarPrestonSurface
from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.assess_feed_rpm_ppo import eligible


class RPMTests(unittest.TestCase):
    def test_reduced_average_error_cannot_hide_a_larger_positive_spike(self):
        parent = {'contact_cv': .455, 'temporal_rate_cv': .03, 'spike_relative_rmse': .17,
            'windows': {key: {'rate_relative_rmse': .04} for key in ('path85_90pct', 'path98p5_100pct')}}
        candidate = {'completed': True, 'fault_samples': 0, 'mean_rate_ratio': 1.,
            'contact_cv': .454, 'temporal_rate_cv': .01, 'spike_relative_rmse': .06,
            'maximum_positive_rate_error': .42,
            'windows': {key: {'rate_relative_rmse': .003} for key in ('path85_90pct', 'path98p5_100pct')}}
        rules = {'fault_samples': 0, 'mean_rate_ratio_min': .97, 'mean_rate_ratio_max': 1.03,
            'spike_rate_rmse_ratio_max': .95, 'temporal_rate_cv_ratio_max': .95,
            'full_contact_spatial_cv_ratio_max': 1.01, 'maximum_positive_rate_error_max': .1}
        checks = eligible(parent, candidate, rules)
        self.assertTrue(checks['whole_contact_temporal_cv_improved'])
        self.assertFalse(checks['maximum_positive_rate_spike_bounded'])
        candidate['maximum_positive_rate_error'] = .08
        self.assertTrue(all(eligible(parent, candidate, rules).values()))

    def test_causal_lag_slew_and_reset(self):
        c = RPMConfig()
        spindle = VirtualSpindle(c)
        self.assertEqual(spindle.step(1., enabled=False), 1000.)
        first = spindle.step(1.)
        self.assertGreater(first, 1000.)
        self.assertLessEqual(first - 1000., c.slew_rpm_s * .008)
        self.assertLess(first, spindle.commanded_rpm)
        spindle.reset()
        self.assertEqual(spindle.rpm, 1000.)
        self.assertEqual(spindle.commanded_rpm, 1000.)

    def test_permitted_contact_forces_fit_numerical_range(self):
        c = RPMConfig()
        for force in (1.5, 14., 20., 24., 100.):
            ideal = 1000. * 20. / force
            self.assertLess(c.command(-1.), ideal)
            self.assertGreater(c.command(1.), ideal)

    def test_variable_rpm_removes_mrr_spike_feed_alone_cannot(self):
        def rate(force, feed, rpm):
            s = PlanarPrestonSurface([[0., 0., 0.], [10., 0., 0.]],
                                     velocity_model='rotating_disk', spindle_rpm=rpm, cell_size_mm=.5)
            s.deposit([5., 0., 0.], [feed, 0., 0.], force, .008)
            return s.integrated_volume / .008
        base = rate(20., 6., 1000.)
        self.assertAlmostEqual(rate(24., 6., 1000. * 20. / 24.) / base, 1., places=4)
        self.assertGreater(rate(24., 7.2, 1000.) / base, 1.19)
        self.assertGreater(rate(24., 4., 1000.) / base, 1.19)

    def test_fixed_target_prevents_zero_removal_uniformity_loophole(self):
        c = RPMConfig()
        ideal, stopped = RateObjective(c), RateObjective(c)
        for _ in range(100):
            good = ideal.update(c.target_rate_over_k, 20., 0.)
            bad = stopped.update(0., 20., 0.)
        self.assertGreater(good, bad)
        self.assertAlmostEqual(ideal.metrics()['mean_rate_ratio'], 1.)
        self.assertEqual(stopped.metrics()['mean_rate_ratio'], 0.)
        self.assertLess(ideal.metrics()['temporal_rate_cv'], 1.e-7)

    def test_target_rate_and_contact_integration(self):
        c = RPMConfig()
        self.assertAlmostEqual(c.target_rate_over_k, 20943.951023931954)
        objective = RateObjective(c)
        objective.update(c.target_rate_over_k * .1, 2., 20.)
        objective.update(c.target_rate_over_k, 20., 0.)
        self.assertEqual(objective.metrics()['contact_samples'], 2)
        self.assertEqual(objective.metrics()['spike_seconds'], .008)
        self.assertAlmostEqual(objective.metrics()['mean_rate_ratio'], .55)
        objective.reset()
        self.assertEqual(objective.metrics()['contact_samples'], 0)

    def test_two_output_export_preserves_actions(self):
        torch.set_num_threads(1)
        model = FeedRPMActorCritic()
        obs = torch.randn(32, 22)
        action, latent, logprob, _, values = model.action_value(obs)
        self.assertEqual(tuple(action.shape), (32, 2))
        self.assertEqual(tuple(logprob.shape), (32,))
        self.assertTrue((action.abs() <= 1.).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actor.pt'
            export_actor(model, path, RPMConfig(), {})
            actor = torch.jit.load(str(path))
            mean = model.actor(obs).tanh()
            expected = torch.stack((.125 + .875 * mean[:, 0], mean[:, 1]), -1)
            torch.testing.assert_close(actor(obs), expected)
        # With constant removal flux, raw force changes must not double-compensate feed.
        obs[:, 19] = 0.
        first = model.actor(obs)[:, 0]
        obs[:, 1] += .2
        second = model.actor(obs)[:, 0]
        torch.testing.assert_close(first, second)


if __name__ == '__main__':
    unittest.main()
