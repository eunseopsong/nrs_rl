"""Prevent old volume constraints from silently returning to model selection."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import ProfileActor, fit_profile
from nrs_rl.tasks.manager_based.nrs_rl.utils.uniformity_priority_metrics import assess_uniformity


class UniformityPriorityTests(unittest.TestCase):
    def setUp(self):
        self.base=dict(spatial_cv=.4,roi_volume_over_k=15000.,time_s=115.,rate_cv=.8,
            force_mean_n=20.,completed=True,fault_fraction=0.)

    def test_volume_duration_and_force_do_not_penalize_uniformity(self):
        for volume,duration,force in [(7500.,150.,10.),(22000.,90.,30.)]:
            candidate={**self.base,'spatial_cv':.32,'roi_volume_over_k':volume,
                'time_s':duration,'force_mean_n':force,'entry_force_peak_n':99.}
            result=assess_uniformity(candidate,self.base)
            self.assertTrue(result['passed'])
            self.assertEqual(result['score'],-.32)
            self.assertAlmostEqual(result['gains']['spatial_cv_gain'],.2)

    def test_empty_invalid_faulted_or_incomplete_run_cannot_win(self):
        for update in [dict(roi_volume_over_k=0.),dict(roi_volume_over_k=float('nan')),
                       dict(spatial_cv=float('nan')),dict(completed=False),dict(fault_fraction=.001)]:
            result=assess_uniformity({**self.base,**update},self.base)
            self.assertFalse(result['passed'])
            self.assertTrue(math.isfinite(result['score']))
            self.assertLess(result['score'],-1.)
        self.assertTrue(assess_uniformity({**self.base,'spatial_cv':0.},self.base)['passed'])

    def test_force_profile_interpolation_and_export_mapping(self):
        for profile,low,high,scale in [([14.,24.,14.],14.,24.,6.),([8.,30.,8.],8.,30.,12.)]:
            actor=ProfileActor(profile,[.9,.03,-.03,.18],force_min=low,force_max=high,force_scale=scale)
            obs=torch.zeros(7,16);obs[:,0]=torch.tensor([-1.,0.,.25,.5,.75,1.,2.])
            expected=torch.tensor([low,low,(low+high)/2.,high,(low+high)/2.,low,low])
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'actor.pt';torch.jit.save(torch.jit.script(actor),str(path))
                action=torch.jit.load(str(path))(obs)
            torch.testing.assert_close(20.+scale*action[:,1],expected)
            self.assertTrue(torch.all(action[:,2]==-1.))
            self.assertLessEqual(float(action.abs().max()),1.)

    def test_profile_learning_flattens_nonuniform_dose_without_volume_target(self):
        matrix=np.diag([1.,1.25,1.5,1.25,1.])
        force,info=fit_profile([matrix],123,bounds=(8.,30.),max_knot_delta=6.)
        initial_cv=np.std(matrix@np.full(5,20.))/np.mean(matrix@np.full(5,20.))
        self.assertLess(info['predicted_cv_by_scenario'][0],initial_cv*.05)
        self.assertTrue(np.all((force>=8.)&(force<=30.)))
        self.assertLessEqual(np.max(np.abs(np.diff(force))),6.+1.e-5)


if __name__=='__main__':unittest.main()
