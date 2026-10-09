"""Fine reward integration must preserve policy inputs and agree with surface physics."""
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.removal_env import RemovalEnv
from nrs_rl.tasks.manager_based.nrs_rl.mdp.removal_reward_surface import RemovalRewardSurface


class FineRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reward_grid_preserves_observations_motion_and_resets(self):
        original = RemovalEnv(3, seed=3501)
        finer = RemovalEnv(3, seed=3501, reward_cell_mm=1.)
        different_reward = False
        for step in range(100):
            action = torch.tensor([-.5, .3, .8]) * (-1 if step % 9 == 0 else 1)
            a, ra, da, _ = original.step(action)
            b, rb, db, _ = finer.step(action)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(da, db)
            for field in ['force', 'cursor', 'speed', 'depth', 'residence']:
                torch.testing.assert_close(getattr(original, field), getattr(finer, field), rtol=0, atol=0)
            different_reward |= not torch.equal(ra, rb)
            if step == 65:
                original.reset(torch.tensor([1]))
                finer.reset(torch.tensor([1]))
                self.assertEqual(float(finer.reward_surface.depth[1].sum()), 0.)
                self.assertGreater(float(finer.reward_surface.depth[0].sum()), 0.)
        self.assertTrue(different_reward)

    def test_half_mm_deposition_matches_independent_rotating_surface(self):
        reward = RemovalRewardSurface(3, 'cpu', .5)
        g = reward.geometry
        reference_arc = np.r_[0., np.linalg.norm(np.diff(g.positions, axis=0), axis=1).cumsum()]
        points = np.stack([np.interp(g.arc, reference_arc, g.positions[:, j]) for j in range(3)], axis=1)
        indices = np.array([117, 501, 1203])
        cursor_mid = torch.tensor(.5 * (g.arc[indices] + g.arc[indices + 1]), dtype=torch.float32)
        previous, cursor = cursor_mid - .24, cursor_mid + .24
        force = torch.tensor([18., 20., 22.])
        reward.deposit(previous, cursor, force, torch.zeros(3))
        self.assertEqual(float(reward.processing.sum()), 0.)
        expected = []
        for i, index in enumerate(indices):
            point = .5 * (points[index] + points[index + 1])
            direction = points[index + 1] - points[index]
            speed = float(cursor[i] - previous[i]) / .08
            velocity = direction / np.linalg.norm(direction) * speed
            g.surface.reset()
            g.surface.deposit(point, velocity, float(force[i]), .08)
            expected.append(g.surface.depth.ravel()[g.roi_flat].copy())
        expected = np.stack(expected)
        np.testing.assert_allclose(reward.depth.numpy(), expected, atol=2.e-5, rtol=2.e-6)
        reward.deposit(previous, cursor, force, torch.full((3,), 2.))
        np.testing.assert_allclose(reward.processing.numpy(), expected, atol=2.e-5, rtol=2.e-6)
        values = reward.processing.double().numpy()[:, reward.mask.numpy()]
        cv, ratio, full_ratio = reward.metrics()
        np.testing.assert_allclose(cv.numpy(), values.std(1) / values.mean(1), rtol=2.e-6)
        np.testing.assert_allclose(ratio.numpy(), values.sum(1) * .25 / float(reward.baseline_processing_volume), rtol=2.e-6)
        np.testing.assert_allclose(full_ratio.numpy(), 2.*expected.sum(1) * .25 / float(reward.baseline_full_volume), rtol=2.e-6)
        final_cursor = torch.full((3,), reward.length)
        torch.testing.assert_close(reward.predicted_cv2(final_cursor), cv.square(), rtol=2.e-6, atol=2.e-6)


if __name__ == '__main__':
    unittest.main()
