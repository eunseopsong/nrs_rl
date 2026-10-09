"""Regression guards for selecting an actual improvement, not reward hacking."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import copy
import unittest

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_refinement import evaluate_candidate, promotion_decision


class RefinementTests(unittest.TestCase):
    def metrics(self):
        base = dict(completed=True, processing_samples=14581, processing_rate_cv=.22,
            force_peak_n=28., force_rmse_n=1.5, force_mean_n=20., cross_track_rmse_mm=2.2,
            endpoint_tangential_error_mm=7.5, measured_path_length_mm=630.,
            fault_fraction=0., shield_fraction=0., preston_surface=dict(
                roi_cells=1000, roi_area_mm2=250., spatial_depth_cv=.52,
                roi_volume_over_k=12500., tcp_full_footprint_volume_over_k=12530., zero_depth_fraction=.08))
        previous = copy.deepcopy(base)
        previous.update(processing_rate_cv=.18, cross_track_rmse_mm=1.9,
                        endpoint_tangential_error_mm=4., measured_path_length_mm=675.)
        previous['preston_surface'].update(spatial_depth_cv=.42, roi_volume_over_k=13400.,
            tcp_full_footprint_volume_over_k=13450., zero_depth_fraction=.03)
        candidate = copy.deepcopy(previous)
        candidate.update(processing_rate_cv=.16, cross_track_rmse_mm=1.8,
                         endpoint_tangential_error_mm=3.8, measured_path_length_mm=681.)
        candidate['preston_surface'].update(spatial_depth_cv=.40, roi_volume_over_k=13500.,
            tcp_full_footprint_volume_over_k=13550., zero_depth_fraction=.025)
        return {'constant_a': base, 'constant_b': copy.deepcopy(base), 'previous': previous}, candidate

    def test_genuine_improvement_passes_with_exact_reward_decomposition(self):
        comparisons, candidate = self.metrics()
        result = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        self.assertTrue(result['eligible'], result['rejections'])
        self.assertAlmostEqual(result['score'], sum(result['weighted_reward'].values())+result['constraint_penalty'])
        self.assertTrue(promotion_decision([{'seed': s, 'assessment': result} for s in (609, 610, 611)])['passed'])

    def test_extra_removal_cannot_buy_excess_force_or_extra_time(self):
        for field, value in [('force_peak_n', 33.), ('force_rmse_n', 1.7), ('processing_samples', 16000)]:
            comparisons, candidate = self.metrics()
            candidate[field] = value
            candidate['preston_surface']['roi_volume_over_k'] = 14000.
            result = evaluate_candidate(candidate, comparisons, 700., full_path=True)
            self.assertFalse(result['eligible'], field)
            self.assertFalse(promotion_decision([{'seed': s, 'assessment': result} for s in (609, 610, 611)])['passed'])

    def test_uniformity_without_volume_improvement_does_not_promote(self):
        comparisons, candidate = self.metrics()
        candidate['preston_surface']['roi_volume_over_k'] = comparisons['previous']['preston_surface']['roi_volume_over_k']
        result = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        self.assertTrue(result['eligible'])
        decision = promotion_decision([{'seed': s, 'assessment': result} for s in (609, 610, 611)])
        self.assertFalse(decision['passed'])
        self.assertIn('Mean ROI volume increase below 0.2% versus previous', decision['rejections'])

    def test_a_failed_condition_cannot_hide_behind_the_mean(self):
        comparisons, candidate = self.metrics()
        good = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        candidate['processing_rate_cv'] = .20
        bad = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        self.assertFalse(promotion_decision([{'seed': 609, 'assessment': good},
            {'seed': 610, 'assessment': good}, {'seed': 611, 'assessment': bad}])['passed'])

    def test_invalid_comparator_rejects_the_candidate(self):
        comparisons, candidate = self.metrics()
        comparisons['previous']['fault_fraction'] = .001
        result = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        self.assertFalse(result['eligible'])
        self.assertIn('previous: incomplete_or_safety_event', result['rejections'])

    def test_artificial_prefix_endpoint_is_not_a_final_endpoint_gate(self):
        comparisons, candidate = self.metrics()
        candidate['endpoint_tangential_error_mm'] = 10.
        prefix = evaluate_candidate(candidate, comparisons, 700., full_path=False)
        self.assertTrue(prefix['eligible'], prefix['rejections'])
        self.assertEqual(prefix['weighted_reward']['endpoint_gain'], 0.)
        full = evaluate_candidate(candidate, comparisons, 700., full_path=True)
        self.assertFalse(full['eligible'])
        self.assertTrue(any('endpoint_error_ratio_max' in r for r in full['rejections']))


if __name__ == '__main__':
    unittest.main()
