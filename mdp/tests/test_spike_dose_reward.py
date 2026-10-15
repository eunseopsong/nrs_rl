"""Behavioral regressions for the causal transient removal reward (no Isaac GUI)."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.spike_dose_reward import (
    DEFAULT_CONFIG, PathAdvance, SpikeDoseConfig, SpikeDoseReward)
from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_latest_force_env import IsaacLatestForceEnv
from nrs_rl.tasks.manager_based.nrs_rl.utils.feed_limits import ForwardSpeedLimiter
from nrs_rl.tasks.manager_based.nrs_rl.utils.preston_surface import PlanarPrestonSurface


DT = .008


def warm_tracker():
    tracker = SpikeDoseReward(SpikeDoseConfig.load(DEFAULT_CONFIG))
    for _ in range(125):
        tracker.update(20., 20000. * DT, 6. * DT)
    return tracker


def response_trial(force_after=23., extra_delay_s=0., hold=False, reverse=False):
    """Synthetic physical feed response through the unchanged deployed limits.

    Force is prescribed, not an Isaac prediction. Use the 80 ms policy hold,
    80 ms action lowpass, 18 mm/s² action slew, and 16/160 feed limits.
    """
    tracker = warm_tracker()
    limiter = ForwardSpeedLimiter(12.)
    for _ in range(1000):
        limiter.step(6.)
    requested = filtered = 6.
    speeds, rewards = [], []
    for i in range(250):
        # The first observation of the step is available on the next policy tick.
        if i % 10 == 0 and i * DT + 1.e-12 >= .080 + extra_delay_s and not hold:
            ratio = force_after / 20.
            requested = 6. / ratio if reverse else 6. * ratio
        filtered += np.clip((1. - np.exp(-DT / .080)) * (requested - filtered),
                            -18. * DT, 18. * DT)
        speed = limiter.step(filtered)
        result = tracker.update(force_after, 1000. * force_after * DT, speed * DT)
        speeds.append(speed)
        rewards.append(result.reward)
    return tracker.metrics(), np.array(speeds), np.array(rewards)


class SpikeDoseTests(unittest.TestCase):
    def test_normal_cv_feed_changes_and_small_noise_have_no_event_cost(self):
        tracker = warm_tracker()
        for i in range(1000):
            force = 20. + .05 * np.sin(i * .05)
            sample = tracker.update(force, 1000. * force * DT, (6. + 3. * np.sin(i * .04)) * DT)
            self.assertEqual(sample.reward, 0.)
        self.assertEqual(tracker.metrics()['event_count'], 0)

    def test_measured_compensation_beats_hold_and_wrong_direction_both_signs(self):
        for force in (17., 23.):
            fast, speeds, _ = response_trial(force)
            hold, _, _ = response_trial(force, hold=True)
            wrong, _, _ = response_trial(force, reverse=True)
            self.assertGreater(fast['reward'], hold['reward'])
            self.assertGreater(hold['reward'], wrong['reward'])
            self.assertLess(fast['event_abs_error_seconds'], hold['event_abs_error_seconds'])
            np.testing.assert_allclose(speeds[-1], 6. * force / 20., atol=2.e-4)

    def test_fast_response_beats_delayed_without_relaxing_motion_limits(self):
        fast, speeds, _ = response_trial()
        delayed, _, _ = response_trial(extra_delay_s=.32)
        self.assertGreater(fast['reward'], delayed['reward'])
        acc = np.diff(np.r_[6., speeds]) / DT
        jerk = np.diff(np.r_[0., acc]) / DT
        self.assertLessEqual(abs(acc).max(), 16. + 1.e-8)
        self.assertLessEqual(abs(jerk).max(), 160. + 1.e-6)
        self.assertGreaterEqual(speeds.min(), 0.)
        self.assertLessEqual(speeds.max(), 12.)

    def test_persistent_force_step_cannot_adapt_away_its_loss(self):
        tracker = warm_tracker()
        dose_before = tracker.reference_rate / tracker.reference_feed
        for _ in range(1500):
            last = tracker.update(23., 23000. * DT, 6. * DT)
        self.assertTrue(last.event)
        self.assertAlmostEqual(last.error, .15)
        self.assertEqual(dose_before, tracker.reference_rate / tracker.reference_feed)
        self.assertLess(last.reward, 0.)

    def test_stationary_jitter_contact_loss_and_shield_cannot_erase_error(self):
        for invalid in ({'contact': False}, {'shield': True}):
            tracker = warm_tracker()
            for _ in range(20):
                tracker.update(23., 23000. * DT, 6. * DT)
            dose_before = tracker.reference_rate / tracker.reference_feed
            for i in range(40):
                sample = tracker.update(23., 23000. * DT, .05 * (-1)**i)
            self.assertTrue(sample.event)
            self.assertGreater(sample.error, 1.)
            self.assertLess(sample.reward, 0.)
            invalid_sample = tracker.update(0., 0., 0., **invalid)
            self.assertLessEqual(invalid_sample.reward, sample.reward)
            self.assertTrue(tracker.event)
            self.assertEqual(dose_before, tracker.reference_rate / tracker.reference_feed)
            self.assertGreater(tracker.update(23., 23000. * DT, 0.).error, 1.)

    def test_reset_isolation_and_recovery(self):
        a, b = warm_tracker(), warm_tracker()
        for _ in range(60):
            a.update(23., 23000. * DT, 6. * DT)
        a.reset()
        self.assertEqual(a.metrics(), SpikeDoseReward().metrics())
        self.assertTrue(b.metrics()['reference_established'])
        for _ in range(60):
            b.update(23., 23000. * DT, 6. * DT)
        for _ in range(200):
            b.update(20., 20000. * DT, 6. * DT)
        self.assertFalse(b.event)
        self.assertEqual(b.metrics()['event_count'], 1)

    def test_projection_uses_measured_signed_motion_including_duplicate_waypoints(self):
        advance = PathAdvance([[0, 0, 0], [2, 0, 0], [2, 0, 0], [2, 3, 0]])
        self.assertEqual(advance.project([0., 0., 3.], 1.), 0.)
        self.assertEqual(advance.project([-.1, 0., 0.], 1.), -.1)
        self.assertEqual(advance.project([0., .1, 0.], 4.), .1)
        self.assertEqual(advance.project([0., 0., 0.], 400.), 0.)

    def test_actual_rotary_model_requires_faster_feed_for_positive_force_spike(self):
        surface = PlanarPrestonSurface([[0., 0., 0.], [20., 0., 0.]], cell_size_mm=.5,
                                       velocity_model='rotating_disk', spindle_rpm=1000.)
        def rate(force, feed):
            surface.reset()
            surface.deposit([10., 0., 0.], [feed, 0., 0.], force, DT)
            return surface.integrated_volume / DT
        nominal = rate(20., 6.) / 6.
        for force in (17., 23.):
            compensated = rate(force, 6. * force / 20.) / (6. * force / 20.)
            hold = rate(force, 6.) / 6.
            self.assertLess(abs(compensated / nominal - 1.), 1.e-4)
            self.assertAlmostEqual(abs(hold / nominal - 1.), .15)
        self.assertLess(abs(rate(20., 12.) / rate(20., 6.) - 1.), 1.e-4)
        self.assertGreater(rate(20., 0.), 0.)

    def test_nonfinite_and_unsafe_config_rejected(self):
        for values in ((float('nan'), 1., 0.), (20., -1., 0.), (20., 1., float('inf'))):
            with self.assertRaises(ValueError):
                warm_tracker().update(*values)
        for kwargs in ({'sample_dt_s': .01}, {'invalid_contact_loss_per_s': 1.},
                       {'force_release_n': 2.}, {'window_s': .081}):
            with self.assertRaises(ValueError):
                SpikeDoseConfig(**kwargs)


class ActionRepeatIntegrationTests(unittest.TestCase):
    def test_capture_uses_measured_motion_and_preserves_surface_integration(self):
        reference = np.array([[0., 0., 0.], [20., 0., 0.]])
        def surface():
            return PlanarPrestonSurface(reference, cell_size_mm=1.,
                                        velocity_model='rotating_disk', spindle_rpm=1000.)
        env = IsaacLatestForceEnv.__new__(IsaacLatestForceEnv)
        env.previous = [np.array([0., 0., 0.])]
        env.surface, env.fine = [surface()], [surface()]
        env.residence, env.entry_depth = [np.zeros(env.surface[0].shape)], [np.zeros(env.fine[0].shape)]
        env.samples = np.zeros(1, dtype=int)
        env.path_advance = PathAdvance(reference)
        env.spike_rewards = [SpikeDoseReward()]
        env.spike_step_reward = np.zeros(1)
        env.spike_step_event = np.zeros(1, dtype=bool)
        env.spike_collect = np.ones(1, dtype=bool)
        env.term = SimpleNamespace(polishing_active=[True], path_done=[False],
                                   safety_fault_reason=[0], safety_shield_active=[False],
                                   current_sliding_velocity_mm_s=[6.], path_cursor_mm=[0.])
        independent = surface()
        for i in range(160):
            # After the force spike, the scheduled cursor keeps advancing but
            # measured TCP is stationary: commanded-only compensation must fail.
            moving = i < 125
            force = 20. if moving else 23.
            point = torch.tensor([min(i + 1, 125) * 6. * DT, 0., 0.], dtype=torch.float64)
            delta = point.numpy() - env.previous[0]
            velocity = delta / max(np.linalg.norm(delta), 1.e-12) * 6.
            independent.deposit(point.numpy() - .5 * delta, velocity, force, DT)
            env.term.path_cursor_mm[0] = (i + 1) * 6. * DT
            env.capture(0, point, force)
        np.testing.assert_array_equal(env.fine[0].depth, independent.depth)
        self.assertTrue(env.spike_step_event[0])
        self.assertLess(env.spike_step_reward[0], 0.)
        self.assertGreater(env.spike_rewards[0].metrics()['event_peak_abs_error'], 1.)

    def test_terminal_spike_reward_counted_once_and_other_environment_continues(self):
        """Exercise the real wrapper step with deterministic fake physics callbacks."""
        env = IsaacLatestForceEnv.__new__(IsaacLatestForceEnv)
        env.count = 2
        env.potential = env.previous_action = np.zeros(2)
        env.episode_reward = np.array([5., 7.])
        env.spike_rewards = [warm_tracker(), warm_tracker()]
        env.spike_step_reward = np.zeros(2)
        env.spike_step_event = np.zeros(2, dtype=bool)
        env.spike_collect = np.zeros(2, dtype=bool)
        env.term = SimpleNamespace(safety_shield_active=np.zeros(2, dtype=bool))
        env.raw = SimpleNamespace(termination_manager=SimpleNamespace(get_term=lambda _: [True, False]))
        env.predicted = lambda *args, **kwargs: 0.
        env.observation = lambda: torch.zeros(2, 16)
        env.metrics = lambda i, completed: {'volume_ratio': 1., 'full_ratio': 1., 'completed': completed,
                                          'spike_dose': env.spike_rewards[i].metrics()}
        def clear(i):
            env.spike_rewards[i].reset()
            env.episode_reward[i] = 0.
        env.clear = clear
        tick = 0
        def physics(_):
            nonlocal tick
            tick += 1
            for i in range(2):
                env.record_spike_reward(i, 23., 23000. * DT, 6. * DT, True)
            return None, None, torch.tensor([tick == 4, False]), torch.zeros(2, dtype=torch.bool), None
        env.env = SimpleNamespace(step=physics)
        _, reward, done, records = env.step(torch.zeros(2))
        self.assertEqual(done.tolist(), [True, False])
        self.assertAlmostEqual(records[0]['episode_reward'], 5. + float(reward[0]), places=6)
        self.assertAlmostEqual(float(reward[0]), records[0]['spike_dose']['reward'] - .000008, places=7)
        self.assertAlmostEqual(float(reward[1]), env.spike_rewards[1].metrics()['reward'] - .000008, places=7)
        self.assertEqual(env.spike_rewards[0].metrics()['event_seconds'], 0.)
        self.assertGreater(env.spike_rewards[1].metrics()['event_seconds'], 0.)
        # The same force history ending in failure receives the stronger
        # completion penalty, so ending the event is not treated as success.
        env.raw.termination_manager.get_term = lambda _: [False, False]
        env.spike_rewards[0] = warm_tracker()
        env.episode_reward[0] = 5.
        tick = 0
        _, failed_reward, failed_done, failed_records = env.step(torch.zeros(2))
        self.assertTrue(bool(failed_done[0]))
        self.assertFalse(failed_records[0]['completed'])
        self.assertAlmostEqual(float(failed_reward[0] - reward[0]), -100., places=5)


if __name__ == '__main__':
    unittest.main()
