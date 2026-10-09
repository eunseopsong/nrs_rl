"""Control-rate and per-environment autoreset checks without launching Isaac."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import importlib.util
from pathlib import Path
import unittest

import gymnasium as gym
import torch

ROOT = _REPO_ROOT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repeat_module = load("repeat_control", ROOT / "source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/control_action_repeat.py")
worker = load("repeat_worker", ROOT / "scripts/skrl/ppo_policy_node.py")


class FakeControl(gym.Env):
    def __init__(self, finish_at=None):
        self.ticks = 0
        self.finish_at = finish_at
        self.actions = []

    def step(self, actions):
        self.actions.append(actions.clone())
        self.ticks += 1
        done = torch.tensor([self.ticks == self.finish_at, False])
        reward = torch.tensor([1000. if self.finish_at and self.ticks > self.finish_at else 1., 2.])
        obs = {"policy": torch.full((2, 14), float(self.ticks))}
        return obs, reward, done, torch.zeros(2, dtype=torch.bool), {"tick": self.ticks}


class RepeatTests(unittest.TestCase):
    @staticmethod
    def ppo_config():
        return {"agent": {"rollouts": 512, "discount_factor": .999, "lambda": .99,
                          "experiment": {"write_interval": 1000, "checkpoint_interval": 4096}},
                "trainer": {"timesteps": 1000000}}

    def test_default_125hz_preserves_configuration_and_counts_every_tick(self):
        base = self.ppo_config()
        self.assertEqual(repeat_module.ppo_config_for_action_repeat(base, 1), base)
        env = FakeControl(finish_at=2)
        repeated = repeat_module.ControlActionRepeat(env, 1, .999)
        for _ in range(2):
            repeated.step(torch.ones((2, 1)))
        self.assertEqual(repeated.coverage()["control_steps_per_environment"], 2)
        self.assertEqual(repeated.coverage()["finished_episodes"], 1)

    def test_action_hold_preserves_physical_horizon_without_mutating_base(self):
        base = self.ppo_config()
        actual = repeat_module.ppo_config_for_action_repeat(base, 4)
        self.assertEqual(actual["agent"]["rollouts"] * 4, 512)
        self.assertEqual(actual["trainer"]["timesteps"] * 4, 1000000)
        self.assertEqual(actual["agent"]["experiment"]["write_interval"], 250)
        self.assertEqual(actual["agent"]["experiment"]["checkpoint_interval"], 1024)
        self.assertAlmostEqual(actual["agent"]["discount_factor"] ** 250, .999 ** 1000)
        self.assertAlmostEqual(actual["agent"]["lambda"] ** 250, .99 ** 1000)
        self.assertEqual(base, self.ppo_config())

    def test_fractional_counts_round_up_and_invalid_repeats_are_rejected(self):
        base = self.ppo_config()
        base["agent"]["experiment"]["write_interval"] = 0
        base["agent"]["experiment"]["checkpoint_interval"] = -1
        actual = repeat_module.ppo_config_for_action_repeat(base, 3)
        self.assertEqual(actual["agent"]["rollouts"], 171)
        self.assertEqual(actual["agent"]["experiment"], base["agent"]["experiment"])
        for invalid in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                repeat_module.ppo_config_for_action_repeat(base, invalid)

    def test_all_raw_rewards_and_control_ticks_are_preserved(self):
        env = FakeControl()
        repeated = repeat_module.ControlActionRepeat(env, 4, .999 ** 4)
        action = torch.tensor([[.3], [-.2]])
        obs, reward, terminated, truncated, _ = repeated.step(action)
        self.assertEqual(env.ticks, 4)
        torch.testing.assert_close(reward, torch.tensor([1., 2.]) * sum(.999 ** k for k in range(4)))
        self.assertFalse((terminated | truncated).any())
        self.assertTrue((obs["policy"] == 4).all())
        for value in env.actions:
            torch.testing.assert_close(value, action)

    def test_partial_autoreset_cannot_leak_next_episode_reward(self):
        env = FakeControl(finish_at=2)
        repeated = repeat_module.ControlActionRepeat(env, 4, 1.)
        obs, reward, terminated, truncated, info = repeated.step(torch.ones((2, 1)))
        torch.testing.assert_close(reward, torch.tensor([2., 8.]))
        self.assertEqual(terminated.tolist(), [True, False])
        self.assertFalse(truncated.any())
        self.assertTrue((obs["policy"] == 4).all())
        self.assertEqual(info["tick"], 2)
        self.assertEqual(repeated.coverage()["finished_episodes"], 1)
        self.assertEqual(env.actions[2].flatten().tolist(), [0., 1.])
        self.assertEqual(env.actions[3].flatten().tolist(), [0., 1.])

    def test_inference_hold_handles_skipped_observations_and_reset(self):
        class Agent:
            calls = 0
            def act(self, observation, **kwargs):
                self.calls += 1
                return None, None, {"mean_actions": observation.clone()}
        agent = Agent()
        hold = worker.PolicyActionHold(agent, 4)
        result = [float(hold.act(torch.tensor(float(seq)), seq)) for seq in [1, 2, 4, 5, 9, 10, 0]]
        self.assertEqual(result, [1., 1., 1., 5., 9., 9., 0.])
        self.assertEqual(agent.calls, 4)
        hold.reset()
        self.assertEqual(float(hold.act(torch.tensor(20.), 1)), 20.)


if __name__ == "__main__":
    unittest.main()
