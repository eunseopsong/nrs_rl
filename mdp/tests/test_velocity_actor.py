"""CPU checks for the gSDE actor's normalization and 125 Hz inference contract."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = _REPO_ROOT


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load('velocity_worker', ROOT / 'scripts/skrl/ppo_policy_node.py')
export = load('velocity_export', ROOT / 'scripts/skrl/velocity_actor.py')


class ProbeEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(14, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(14, dtype=np.float32), 0., False, False, {}


class ActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = VecNormalize(DummyVecEnv([ProbeEnv]), norm_reward=False, clip_obs=5.)
        self.model = PPO('MlpPolicy', self.env, n_steps=4, batch_size=4, use_sde=True,
                         sde_sample_freq=32, device='cpu', seed=11,
                         policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                            activation_fn=torch.nn.Tanh, log_std_init=-3., use_expln=True))
        self.env.obs_rms.mean = np.linspace(-1, 1, 14)
        self.env.obs_rms.var = np.linspace(.03, 3., 14)
        self.env.obs_rms.count = 1234.
        self.contract = {**worker.SCHEDULER_CONTRACT, 'schema_version': 3, 'observation_size': 14,
                         'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
                         'physics_tool_diameter_mm': 30., 'policy_backend': 'torchscript_sb3_ppo'}

    def tearDown(self):
        self.env.close()

    def save(self, root, observations):
        root = Path(root)
        (root / 'params').mkdir()
        (root / 'params/velocity_policy.json').write_text(json.dumps(self.contract))
        checkpoint = root / 'checkpoints/actor.pt'
        result = export.export_actor(self.model.policy, self.env, checkpoint, self.contract, observations)
        return checkpoint, result

    def test_round_trip_preserves_normalization_clipping_and_raw_mean(self):
        observations = np.random.default_rng(13).normal(size=(64, 14)).astype(np.float32)
        observations[0] = 1e5
        observations[1] = -1e5
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, result = self.save(directory, observations)
            agent, _ = worker._build_agent(str(checkpoint))
            with torch.inference_mode():
                actual = agent.act(torch.from_numpy(observations))[-1]['mean_actions']
                expected = self.model.policy.get_distribution(torch.as_tensor(
                    self.env.normalize_obs(observations))).get_actions(deterministic=True)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-6, rtol=0)
            self.assertLess(result['maximum_mean_action_error'], 2e-6)
            self.assertEqual(float(agent._state_preprocessor.current_count), 1234.)

    def test_worker_recomputes_mean_on_every_control_request(self):
        observations = np.vstack((np.zeros(14), np.ones(14))).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.save(directory, observations)
            agent, _ = worker._build_agent(str(checkpoint))
            hold = worker.PolicyActionHold(agent, 1)
            with torch.inference_mode():
                a = hold.act(torch.tensor(observations[:1]), 0).clone()
                b = hold.act(torch.tensor(observations[1:]), 1).clone()
                repeated = hold.act(torch.tensor(observations[:1]), 2).clone()
            self.assertGreater(float((a-b).abs().max()), 1e-6)
            torch.testing.assert_close(a, repeated)

    def test_embedded_contract_rejects_mismatched_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.save(directory, np.zeros((1, 14), dtype=np.float32))
            self.contract['nominal_speed_mm_s'] = 7.
            (Path(directory) / 'params/velocity_policy.json').write_text(json.dumps(self.contract))
            with self.assertRaisesRegex(ValueError, 'contracts differ'):
                worker._build_agent(str(checkpoint))

    def test_persistent_exploration_weights_do_not_hold_the_state_response(self):
        policy = self.model.policy
        a, b = torch.ones((1, 14)), torch.ones((1, 14)) * -.5
        policy.reset_noise(1)
        with torch.inference_mode():
            first = policy(a)[0]
            same = policy(a)[0]
            changed_state = policy(b)[0]
            policy.reset_noise(1)
            changed_noise = policy(a)[0]
        torch.testing.assert_close(first, same)
        self.assertGreater(float((first-changed_state).abs().max()), 1e-6)
        self.assertGreater(float((first-changed_noise).abs().max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
