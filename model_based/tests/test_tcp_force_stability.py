"""Signed feedback and conservative replicate selection for the follow-up study."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import copy
import unittest

import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_force_stability import rank_family
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import SignedTcpRemovalActor, TcpRemovalActor


class ForceStabilityTests(unittest.TestCase):
    def baseline(self):
        return dict(completed=True, processing_samples=14581, processing_rate_cv=.22,
            force_peak_n=28., force_rmse_n=1.5, force_mean_n=20., cross_track_rmse_mm=2.2,
            endpoint_tangential_error_mm=7.5, measured_path_length_mm=630.,
            fault_fraction=0., shield_fraction=0., preston_surface=dict(
                roi_cells=1000, roi_area_mm2=250., spatial_depth_cv=.52,
                roi_volume_over_k=12800., tcp_full_footprint_volume_over_k=12830., zero_depth_fraction=.08))

    def calibrated(self):
        result=self.baseline()
        result.update(processing_rate_cv=.18, cross_track_rmse_mm=1.9,
                      endpoint_tangential_error_mm=4., measured_path_length_mm=675.)
        result['preston_surface'].update(spatial_depth_cv=.42, roi_volume_over_k=13800.,
            tcp_full_footprint_volume_over_k=13830., zero_depth_fraction=.03)
        return result

    def test_signed_actions_and_scripted_roundtrip(self):
        observations=torch.zeros(3,14)
        observations[:,1]=torch.tensor([.9,1.,1.1])
        observations[:,10]=1.
        actor=SignedTcpRemovalActor([-.14,.013,0.,0.,0.,0.])
        actions=actor(observations)
        self.assertLess(actions[0],0)
        self.assertEqual(actions[1],0)
        self.assertGreater(actions[2],0)
        torch.testing.assert_close(torch.jit.script(actor)(observations),actions)
        observations[:,11]=1.
        self.assertEqual(float(actor(observations).abs().max()),0.)
        with self.assertRaises(ValueError):
            TcpRemovalActor([-.14,0.,0.,0.,0.,0.])

    def test_material_gain_cannot_override_force_stability_eligibility(self):
        candidate=self.calibrated()
        candidate['force_rmse_n']=1.55  # Original +10% gate passes, new +2% condition fails.
        result=rank_family({'candidate':candidate},['candidate'],[self.baseline()],
                           [self.calibrated()],700.)
        self.assertTrue(result['replicates'][0]['base_assessment']['passed'])
        self.assertFalse(result['eligible'])

    def test_adaptive_feedback_must_meet_both_calibrated_replicates(self):
        easy=self.calibrated(); strict=self.calibrated(); strict['force_rmse_n']=1.4
        candidate=self.calibrated();candidate['force_rmse_n']=1.44
        self.assertTrue(rank_family({'c':candidate},['c'],[self.baseline()],[easy],700.)['eligible'])
        self.assertFalse(rank_family({'c':candidate},['c'],[self.baseline()],[easy,strict],700.)['eligible'])

    def test_zero_feedback_uses_both_replicates_and_rejects_a_failed_one(self):
        a,b=self.calibrated(),self.calibrated()
        result=rank_family({'a':a,'b':b},['a','b'],[self.baseline()],[a,b],700.,constant=True)
        self.assertTrue(result['eligible'])
        b['force_peak_n']=33.
        self.assertFalse(rank_family({'a':a,'b':b},['a','b'],[self.baseline()],[a,b],700.,constant=True)['eligible'])


if __name__=='__main__':
    unittest.main()
