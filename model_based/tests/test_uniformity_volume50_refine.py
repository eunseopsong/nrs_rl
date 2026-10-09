# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import unittest
import numpy as np
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50_refine import counterfactual_time, nominal_speed, response_irregular
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import force_filter
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_volume50_refine import refinement_guard


class TimingRefinementTests(unittest.TestCase):
    def test_promotion_needs_volume_and_repeatable_saved_model_advantage(self):
        baseline=dict(eligible=True,spatial_cv=.4,roi_volume_over_k=100.,full_roi_volume_over_k=110.)
        saved=dict(baseline,spatial_cv=.4*(1.-.13),roi_volume_over_k=56.,full_roi_volume_over_k=63.)
        candidate=dict(saved,spatial_cv=.4*(1.-.1311),roi_volume_over_k=53.,full_roi_volume_over_k=60.)
        self.assertTrue(refinement_guard(candidate,baseline,saved)['passed'])
        self.assertFalse(refinement_guard(dict(candidate,spatial_cv=.4*(1.-.1305)),baseline,saved)['passed'])
        self.assertFalse(refinement_guard(dict(candidate,roi_volume_over_k=49.),baseline,saved)['passed'])

    def test_irregular_response_matches_regular_filter_and_is_causal(self):
        t=np.arange(500)*.008
        values=np.column_stack([np.where(t>=1.,20.,8.),np.ones(len(t))])
        actual=response_irregular(values,t,.16,.52)
        np.testing.assert_allclose(actual,force_filter(values,t,.16,.52),rtol=1.e-12,atol=1.e-12)
        np.testing.assert_allclose(actual[t<1.16,0],8.)
        with self.assertRaises(ValueError):response_irregular(values,np.zeros(len(t)),0.,.52)

    def test_time_warp_preserves_order_and_expected_speed_ratio(self):
        t=np.arange(100)*.008
        d={'time':t,'cursor':np.linspace(0.,10.,len(t)),'target':np.full(len(t),14.)}
        old=np.array([.9,0.,0.,0.]);new=np.array([.45,0.,0.,0.])
        np.testing.assert_allclose(counterfactual_time(d,np.full(5,14.),10.,old,old),t)
        warped=counterfactual_time(d,np.full(5,14.),10.,old,new)
        self.assertTrue(np.all(np.diff(warped)>0.))
        self.assertAlmostEqual(warped[-1]/t[-1],float(nominal_speed(14.,old)/nominal_speed(14.,new)))


if __name__=='__main__':unittest.main()
