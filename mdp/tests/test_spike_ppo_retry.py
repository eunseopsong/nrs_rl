"""Behavioral checks for the retraining retry, without an Isaac process."""
import copy
import unittest

import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.ppo_network import PPOActorCritic
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.spike_ppo_auxiliary import (
    clip_actor_critic, compensation_loss, feed, force_observation)
from nrs_rl.tasks.manager_based.nrs_rl.mdp.spike_dose_reward import SpikeDoseConfig, SpikeDoseReward


def tracker(shape):
    reward = SpikeDoseReward(SpikeDoseConfig(loss_shape=shape, error_weight=4., tail_weight=1.))
    for _ in range(125):
        reward.update(20., 160., .048)
    return reward


class RetryTests(unittest.TestCase):
    def test_large_error_reduction_remains_rewarded_below_safe_bound(self):
        rewards = []
        for error in (1., 2., 3., 4.):
            reward = tracker('bounded_pseudo_huber')
            # Prescribed measured displacement; positive excess speed produces
            # negative local dose error. Test the absolute tail in both cases.
            for _ in range(100):
                sample = reward.update(23., 184., (1.15 + error) * .048)
            rewards.append(sample.reward)
            self.assertGreater(sample.reward, -.16)
            invalid = reward.update(0., 0., 0., contact=False)
            self.assertLess(invalid.reward, sample.reward)
        self.assertTrue(all(a > b for a, b in zip(rewards, rewards[1:])))

    def test_reward_shape_keeps_detector_and_reported_error_unchanged(self):
        old, new = tracker('clipped_quadratic'), tracker('bounded_pseudo_huber')
        for force, advance in [(23., .09)] * 100 + [(20., .048)] * 150:
            a, b = (x.update(force, force * 8., advance) for x in (old, new))
            self.assertEqual(a.event, b.event)
            self.assertEqual(a.error, b.error)
        for field in ('event_count', 'event_seconds', 'event_abs_error_seconds', 'event_error_rms'):
            self.assertEqual(old.metrics()[field], new.metrics()[field])

    def test_critic_error_cannot_change_independently_clipped_actor_gradient(self):
        torch.manual_seed(11)
        first = PPOActorCritic()
        second = copy.deepcopy(first)
        obs = torch.randn(32, 16)
        for model, scale in [(first, 1.), (second, 1.e6)]:
            (model.actor(obs).square().mean() + scale * model.critic(obs).square().mean()).backward()
            clip_actor_critic(model)
        for a, b in zip(first.actor.parameters(), second.actor.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=0., rtol=0.)

    def test_counterfactual_force_keeps_observation_semantics(self):
        obs = torch.randn(10, 16)
        original = obs.clone()
        changed = force_observation(obs, 1.2)
        torch.testing.assert_close(obs, original)
        torch.testing.assert_close(changed[:, 13], 1.2 * obs[:, 2] - 1.)
        keep = [i for i in range(16) if i not in (1, 13)]
        torch.testing.assert_close(changed[:, keep], obs[:, keep])

    def test_auxiliary_learns_force_response_without_moving_nominal_feed(self):
        class GainActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, obs):
                desired = 6. * (1. + self.gain * (obs[:, 1] - 1.))
                return ((desired - 6.75) / 5.25).clamp(-.9999, .9999).atanh()

        model = GainActor()
        parent = copy.deepcopy(model).requires_grad_(False)
        obs = torch.zeros(64, 16)
        obs[:, 0] = .5
        obs[:, 1] = obs[:, 2] = obs[:, 10] = obs[:, 12] = 1.
        valid = torch.ones(64, dtype=torch.bool)
        optimizer = torch.optim.Adam(model.parameters(), lr=.05)
        initial, _, _ = compensation_loss(model, parent, obs, valid)
        for _ in range(80):
            loss, anchor, count = compensation_loss(model, parent, obs, valid)
            optimizer.zero_grad()
            (loss + anchor).backward()
            optimizer.step()
        final, anchor, count = compensation_loss(model, parent, obs, valid)
        self.assertLess(float(final), float(initial) * .01)
        self.assertEqual(count, 64)
        torch.testing.assert_close(feed(model, obs), torch.full((64,), 6.))
        self.assertIsNone(parent.gain.grad)
        for column in (10, 11):
            invalid = obs.clone()
            invalid[:, column] = 0. if column == 10 else 1.
            force, anchor, count = compensation_loss(model, parent, invalid, valid)
            self.assertEqual(count, 0)
            self.assertEqual(float(force + anchor), 0.)
        self.assertEqual(compensation_loss(model, parent, obs, ~valid)[2], 0)


if __name__ == '__main__':
    unittest.main()
