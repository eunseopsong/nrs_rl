"""Behavioral checks for the bounded feedback family before physics screening."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(_REPO_ROOT / 'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.bounded_velocity_actor import BoundedVelocityActor


class CorrectiveDirectionTests(unittest.TestCase):
    def states(self, count=101):
        obs = torch.zeros(count, 14)
        obs[:, 1], obs[:, 3], obs[:, 6], obs[:, 10] = 1., .5, .5, 1.
        return obs

    def test_rate_deficit_increases_request_and_excess_decreases_it(self):
        obs = self.states()
        obs[:, 8] = torch.linspace(-1, 1, len(obs))
        actions = BoundedVelocityActor([.7, .3, 0., 0.])(obs).squeeze()
        self.assertTrue(torch.all(actions.diff() <= 0))
        self.assertGreater(float(actions[0]), 0)
        self.assertLess(float(actions[-1]), 0)

    def test_force_direction_and_progress_invariance(self):
        actor = BoundedVelocityActor([1., .3, .2, 0.])
        obs = self.states()
        obs[:, 1] = torch.linspace(.8, 1.2, len(obs))
        obs[:, 8] = obs[:, 1] - 1.
        self.assertTrue(torch.all(actor(obs).squeeze().diff() <= 0))
        obs = self.states()
        obs[:, 7], obs[:, 13], obs[:, 5] = torch.rand(101), torch.rand(101), torch.rand(101)
        torch.testing.assert_close(actor(obs), torch.zeros(101, 1))

    def test_bounds_contact_and_script_export_agree(self):
        actor = BoundedVelocityActor([1.5, .8, .6, .08])
        obs = self.states(1024)
        obs[:, :10] = torch.randn(1024, 10) * 100
        obs[0, 10], obs[1, 11] = 0., 1.
        actions = actor(obs)
        self.assertTrue(torch.isfinite(actions).all())
        self.assertLessEqual(float(actions.abs().max()), 1.)
        self.assertEqual(float(actions[0]), 0.)
        self.assertEqual(float(actions[1]), 0.)
        torch.testing.assert_close(torch.jit.script(actor)(obs), actions)

    def test_invalid_feedback_direction_is_rejected(self):
        for parameters in ([-.1, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 0, 1], [0, float('nan'), 0, 0]):
            with self.assertRaises(ValueError):
                BoundedVelocityActor(parameters)


if __name__ == '__main__':
    unittest.main()
