"""Check scalar ABI, bounded preview, response fidelity and promotion gates."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from pathlib import Path
import copy
import json
import tempfile
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_dynamics import (PreviewResidenceActor, export_actor, policy_feed,
                                  rollout_residence, replay_command_response)
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import check_contract
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_dynamics import PREVIOUS, check_preserved, fit_residence_qp
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_fixed_force_dynamics import acceptance


class DynamicsTests(unittest.TestCase):
    def test_convex_residence_fit_and_constraints(self):
        matrix = np.diag([1., 2., 1., 2.])
        profile, metrics = fit_residence_qp([matrix, matrix.copy()], volume_ratio=1.,
            minimum_speed=1.5, maximum_speed=12., knot_delta=.9)
        depth = matrix@profile
        self.assertLess(depth.std()/depth.mean(), .001)
        self.assertLess(abs(depth.mean()/matrix.sum(axis=1).mean()-1.), 1.e-6)
        self.assertLessEqual(np.max(np.abs(np.diff(profile))), .900002)
        self.assertTrue(metrics['solver_success'])

    def test_scalar_export_preview_parity_and_bounds(self):
        q = np.array([1., .5, 4., 1.2])
        obs = torch.zeros((4096, 16)); obs[:, 0] = torch.linspace(-1., 2., 4096)
        for preview in [0., .25, .4]:
            with tempfile.TemporaryDirectory(dir='/tmp') as directory:
                path = Path(directory)/'actor.pt'
                contract = export_actor(path, q.tolist(), path_length_mm=698.,
                    lookahead_time_s=preview, candidate='test', protocol={})
                check_contract(contract)
                with torch.inference_mode(): output = torch.jit.load(str(path))(obs)
                self.assertEqual(tuple(output.shape), (4096, 1))
                self.assertTrue(torch.isfinite(output).all() and (output.abs() <= 1.).all())
                feeds = 6.*(1.+output.numpy().ravel())
                expected = np.array([policy_feed(q, float(s)*698., 698., preview) for s in obs[:, 0]])
                np.testing.assert_allclose(feeds, expected, atol=8.e-6, rtol=2.e-6)
                bad = {**contract, 'target_force_n': 19.}
                with self.assertRaises(ValueError): check_contract(bad)
        with self.assertRaises(ValueError): PreviewResidenceActor(q, 698., .5)

    def test_constant_and_closed_loop_response(self):
        for delay in [0, 3]:
            q, seconds, done = rollout_residence(np.ones(4), 698., 0., delay, 289)
            self.assertTrue(done)
            self.assertLess(abs(seconds-698./6.), 1.)
            np.testing.assert_allclose(q[10:], 1., atol=1.e-7)

    def test_recorded_command_response(self):
        with np.load(PREVIOUS/'physics/selection_b_2701/dense289_110_trace.npz') as data:
            raw = 6.*(1.+data['raw_action'].ravel())
            initial = 6.*(1.+data['filtered_action'].ravel()[0])
            _, estimated = replay_command_response(raw, 1, initial)
            keep = data['time_s'] >= 2.
            error = float(np.max(np.abs(estimated[keep]-data['commanded_speed_mm_s'][keep])))
        self.assertLess(error, .01)

    def test_save_and_performance_gate(self):
        check_preserved()
        data = {'eligible': True, 'spatial_cv': .44, 'mode6_cv_gain': 0.}
        report = {'candidates': {n: copy.copy(data) for n in ['constant', 'constant_b', 'saved', 'selected']}}
        report['candidates']['saved']['mode6_cv_gain'] = .1215
        report['candidates']['selected']['mode6_cv_gain'] = .13
        self.assertTrue(acceptance(report)['passed'])
        report['candidates']['selected']['mode6_cv_gain'] = .123
        self.assertFalse(acceptance(report)['passed'])
        report['candidates']['selected']['mode6_cv_gain'] = .13
        report['candidates']['selected']['eligible'] = False
        self.assertFalse(acceptance(report)['passed'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
