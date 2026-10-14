"""PPO checkpoint identity, observation parity, cadence, and rejection checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT, REFERENCE
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_deployment import SAVE
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_runtime import (
    CHECKPOINT_SHA256, FixedForcePPORuntime)

CHECKPOINT = SAVE / 'candidate/checkpoints/actor.pt'


class PPODeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = FixedForcePPORuntime(CHECKPOINT, REFERENCE, 5)
        cls.constant = FixedForcePPORuntime(CHECKPOINT, REFERENCE, 6)

    def setUp(self):
        self.policy.reset(0)
        self.constant.reset(0)

    def telemetry(self, sequence, epoch=1):
        x, y, z = self.policy.geometry.positions[0]
        cursor = (sequence - 1) * .048
        return [2., epoch, sequence, sequence * .008, cursor, self.policy.geometry.length,
                cursor, 0., 20., 0., sequence * .008, 0., 0., 6., .5,
                x + cursor, y, z, 0., 0., 1., 20., 0.]

    def test_actor_identity_and_4096_observation_bounds(self):
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), CHECKPOINT_SHA256)
        self.assertEqual(self.policy.contract['actor_kind'], 'neural_ppo')
        observations = torch.randn(4096, 16, generator=torch.Generator().manual_seed(123)) * 5
        with torch.inference_mode():
            actions = self.policy.actor(observations)
        self.assertEqual(tuple(actions.shape), (4096, 1))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.all((actions >= -.75) & (actions <= 1.)))

    def test_matched_mode6_and_deterministic_inference_cadence(self):
        first = None
        previous = None
        for seq in range(1, 102):
            a = self.policy.process(self.telemetry(seq))
            b = self.constant.process(self.telemetry(seq))
            np.testing.assert_array_equal(a['observation'], b['observation'])
            np.testing.assert_array_equal(a['actor_action'], b['actor_action'])
            self.assertEqual(a['action'][:2], [1, seq])
            self.assertEqual(b['action'][2:], [0.])
            if (seq - 1) % 10 == 0:
                with torch.inference_mode():
                    direct = self.policy.actor(torch.tensor([a['observation']])).numpy()[0]
                np.testing.assert_array_equal(a['action'][2:], direct)
            else:
                self.assertEqual(a['action'][2:], previous)
            previous = a['action'][2:]
            if first is None:
                first = a
        self.assertEqual(self.policy.decisions, 11)
        self.assertEqual(self.constant.decisions, 11)
        self.assertEqual(self.policy.missing_samples, 0)
        again = self.policy.process(self.telemetry(1, epoch=2))
        np.testing.assert_array_equal(first['observation'], again['observation'])
        self.assertEqual(first['action'][2:], again['action'][2:])

    def test_observations_match_frozen_deployed_math_on_isaac_trace(self):
        spec = importlib.util.spec_from_file_location('frozen_fixed_force',
            ROOT / 'deployment/fixed_force_12p59/fixed_force_policy_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frozen = module.FixedForcePolicyRuntime(ROOT / 'deployment/fixed_force_12p59/checkpoints/actor.pt', REFERENCE)
        trace_path = ROOT / 'logs/experiments/ppo_fixed_force_refinement_20261009_1700/physics/heldout_3351/selected_trace.npz'
        with np.load(trace_path) as archive:
            trace = {key: archive[key] for key in archive.files}
            for index in range(1000):
                seq = index + 1
                cursor = float(trace['cursor_mm'][index])
                v = [2., 1., seq, seq * .008, cursor, self.policy.geometry.length, cursor,
                     0., 20., 0., seq * .008, float(trace['safety_shield_active'][index]),
                     float(trace['safety_fault_reason'][index]), float(trace['measured_speed_mm_s'][index]),
                     float(trace['tracking_error_mm'][index]), *trace['tcp_pose'][index, :3].tolist(),
                     0., 0., 1., float(trace['normal_force_n'][index]), 0.]
                a = self.policy.process(v)
                b = frozen.process(v)
                np.testing.assert_array_equal(a['observation'], b['observation'])

    def test_faults_schema_clock_and_reverse_are_rejected(self):
        cases = {0: 1., 5: 1., 7: .1, 8: 19., 9: 1., 10: .03,
                 12: 4., 21: 100.1, 15: float('nan'), 4: -.1}
        for index, value in cases.items():
            with self.subTest(index=index):
                self.policy.reset(0)
                self.policy.process(self.telemetry(1))
                v = self.telemetry(2)
                v[index] = value
                with self.assertRaises(ValueError):
                    self.policy.process(v)
        self.policy.reset(0)
        with self.assertRaises(ValueError):
            self.policy.process(self.telemetry(2))

    def test_replay_foreign_epoch_and_large_gap_are_rejected(self):
        self.policy.process(self.telemetry(1, epoch=2))
        for value in (self.telemetry(1, epoch=2), self.telemetry(2), self.telemetry(100, epoch=2)):
            with self.assertRaises(ValueError):
                self.policy.process(value)

    def test_checkpoint_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'actor.pt'
            path.write_bytes(CHECKPOINT.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'cv11p13'):
                FixedForcePPORuntime(path, REFERENCE)
        with self.assertRaises(ValueError):
            FixedForcePPORuntime(CHECKPOINT, REFERENCE, 7)


if __name__ == '__main__':
    unittest.main()
