# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import unittest
import numpy as np
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50 import fit_constrained
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_volume_guard import assess_volume_guard


class VolumeConstraintTests(unittest.TestCase):
    def test_better_cv_cannot_buy_a_volume_violation(self):
        baseline = dict(eligible=True, spatial_cv=.4, roi_volume_over_k=100., full_roi_volume_over_k=110.)
        candidate = dict(eligible=True, spatial_cv=.34, roi_volume_over_k=55., full_roi_volume_over_k=60.)
        self.assertTrue(assess_volume_guard(candidate,baseline)['passed'])
        for change in [dict(roi_volume_over_k=49.99), dict(full_roi_volume_over_k=54.99),
                       dict(roi_volume_over_k=float('nan')), dict(eligible=False)]:
            self.assertFalse(assess_volume_guard({**candidate, **change},baseline)['passed'])
        saved = dict(candidate, spatial_cv=.35)
        self.assertTrue(assess_volume_guard(candidate,baseline,saved,minimum_gain=.1114)['passed'])
        self.assertFalse(assess_volume_guard(dict(candidate,spatial_cv=.36),baseline,saved)['passed'])

    def test_constrained_fit_satisfies_every_scenario(self):
        rng = np.random.default_rng(35)
        matrices = [rng.uniform(.001,.03,(80,9)), rng.uniform(.001,.02,(80,9))]
        offsets = [rng.uniform(0,.04,80), rng.uniform(0,.06,80)]
        baselines = [float((a@np.full(9,20.)+b).sum()) for a,b in zip(matrices,offsets)]
        p, result = fit_constrained(matrices,offsets,baselines,1.,np.full(9,7.),.55,3.5)
        self.assertTrue(result['solver_success'])
        self.assertTrue(np.all((p >= 4.-1.e-7)&(p <= 30.+1.e-7)))
        self.assertLessEqual(np.max(np.abs(np.diff(p))),3.5+1.e-6)
        for a,b,base in zip(matrices,offsets,baselines):
            self.assertGreaterEqual(float((a@p+b).sum()/base),.55-1.e-7)


if __name__ == '__main__':
    unittest.main()
