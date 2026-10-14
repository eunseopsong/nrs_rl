"""Streaming 16D observations for the saved cv11p13 neural PPO actor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..removal_config import CONTRACT, check_ppo_or_baseline
from ...utils.rotary_geometry import FixedForceGeometry

CHECKPOINT_SHA256 = '363abffb65b4a6078a63baffa9ac864f0609ede73a55605df09e7e1862aa1d0c'
OBSERVATIONS = CONTRACT['observation_names']
TELEMETRY = ['schema', 'epoch', 'sequence', 'simulation_time_s', 'cursor_mm', 'path_length_mm',
    'frontier_mm', 'reverse_mm', 'target_force_n', 'return_allowed', 'elapsed_s', 'shield', 'fault',
    'actual_speed_mm_s', 'tracking_error_mm', 'tcp_x_mm', 'tcp_y_mm', 'tcp_z_mm',
    'normal_x', 'normal_y', 'normal_z', 'normal_force_n', 'done']


class FixedForcePPORuntime:
    def __init__(self, checkpoint, reference, force_control_mode=5):
        if force_control_mode not in (5, 6):
            raise ValueError('Expected Mode 5 or matched Mode 6')
        self.force_control_mode = force_control_mode
        torch.set_num_threads(1)
        checkpoint, reference = Path(checkpoint), Path(reference)
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.sha256 != CHECKPOINT_SHA256:
            raise ValueError('This runtime requires the saved cv11p13 neural PPO checkpoint')
        extra = {'fixed_force_policy.json': ''}
        self.actor = torch.jit.load(str(checkpoint), map_location='cpu', _extra_files=extra).eval()
        self.contract = json.loads(extra['fixed_force_policy.json'])
        check_ppo_or_baseline(self.contract)
        if (self.contract['actor_kind'] != 'neural_ppo'
                or self.contract['source_path_sha256'] != hashlib.sha256(reference.read_bytes()).hexdigest()):
            raise ValueError('PPO checkpoint/reference mismatch')
        self.geometry = FixedForceGeometry(reference)
        self.surface = self.geometry.surface
        self.reset(0)

    def reset(self, epoch):
        self.epoch = epoch
        self.sequence = 0
        self.elapsed = 0.
        self.cursor = 0.
        self.previous_position = None
        self.surface.reset()
        self.residence = np.zeros_like(self.surface.depth)
        self.action = np.zeros(1, dtype=np.float32)
        self.actor_action = self.action.copy()
        self.observation = np.zeros(16, dtype=np.float32)
        self.samples = self.decisions = self.missing_samples = 0

    def process(self, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (23,) or not np.isfinite(v).all() or v[0] != 2.:
            raise ValueError('Expected finite 23D fixed-force telemetry schema 2')
        epoch, sequence = int(v[1]), int(v[2])
        if epoch < 1 or sequence < 1 or v[1] != epoch or v[2] != sequence:
            raise ValueError('Invalid trajectory epoch/sequence')
        if epoch < self.epoch:
            raise ValueError('Old trajectory telemetry rejected')
        if epoch != self.epoch:
            self.reset(epoch)
        if sequence <= self.sequence:
            raise ValueError('Repeated or reordered telemetry rejected')
        if abs(v[5] - self.geometry.length) > .05:
            raise ValueError('Runtime path length differs from trained path')
        if v[12] != 0 or not 0. <= v[21] <= 100.:
            raise ValueError('Controller fault or invalid/over-limit contact force')
        if v[8] != 20. or v[7] != 0. or v[9] != 0.:
            raise ValueError('Force target or forward-only contract violated')
        if self.sequence == 0 and (sequence != 1 or abs(v[10] - .008) > 1.e-8):
            raise ValueError('Missed trajectory start; restart launch')
        if self.sequence and v[4] < self.cursor - 1.e-6:
            raise ValueError('Backward cursor forbidden')
        dt = v[10] - self.elapsed
        ticks = sequence - self.sequence
        if not 0 < dt <= .25 or abs(dt - .008 * ticks) > 1.e-8:
            raise ValueError('Missing or inconsistent control ticks')
        gap = ticks - 1
        self.missing_samples += gap
        # Match the float32 midpoint arithmetic in the independent Isaac evaluator.
        position = v[15:18].astype(np.float32)
        previous = self.previous_position
        self.previous_position = position.copy()
        delta = np.zeros(3) if previous is None else position - previous
        delta[2] = 0.
        norm = np.linalg.norm(delta)
        speed = v[13] if gap == 0 else norm / dt
        velocity = delta / max(norm, 1.e-12) * np.float32(speed)
        midpoint = position - .5 * delta
        contact = v[21] >= 1.5
        self.surface.deposit(midpoint, velocity, v[21], dt, contact=contact)
        if contact:
            center = (midpoint - self.surface.origin) @ self.surface.basis.T
            rows, cols, _, _, weights, _ = self.surface._footprint(center)
            self.residence[rows, cols] += dt * (weights > 0)
        g = self.geometry
        features = g.features(self.surface.depth.ravel()[g.roi_flat][None, :],
            self.residence.ravel()[g.roi_flat][None, :], np.array([v[4]]), np.array([v[6]]),
            np.array([v[21]]), np.array([v[13]]), np.array([v[10]]), np.array([v[7]]),
            np.array([v[8]]), np.array([v[9]]), tracking=np.array([v[14]]), shield=np.array([v[11]]))
        if features.shape != (1, 16) or not np.isfinite(features).all():
            raise ValueError('Invalid PPO observation')
        self.observation = features[0]
        decision = self.sequence == 0 or (sequence - 1) // 10 != (self.sequence - 1) // 10
        if decision:
            with torch.inference_mode():
                proposed = self.actor(torch.from_numpy(features))[0].numpy().copy()
            if (proposed.shape != (1,) or not np.isfinite(proposed).all()
                    or proposed[0] < -.750001 or proposed[0] > 1.000001):
                raise ValueError('PPO output violates the 1.5–12 mm/s feed contract')
            self.actor_action = np.clip(proposed, -.75, 1.)
            self.action = self.actor_action.copy() if self.force_control_mode == 5 else np.zeros(1, dtype=np.float32)
            self.decisions += 1
        self.sequence = sequence
        self.elapsed = v[10]
        self.cursor = v[4]
        self.samples += 1
        return {'action': [epoch, sequence, *self.action.tolist()],
                'observation': self.observation.tolist(), 'decision': decision,
                'missing_samples': self.missing_samples, 'actor_action': self.actor_action.tolist()}
