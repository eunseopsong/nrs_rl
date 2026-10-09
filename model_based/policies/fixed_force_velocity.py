"""One-output velocity policy with fixed force and an assumed virtual spindle.

URDF/joint dynamics are unchanged. Deposition uses the sum of measured feed
and the local velocity of a virtual disk spinning at a constant supplied RPM.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import OBSERVATIONS, REFERENCE, DoseGeometry, PlanarPrestonSurface

SPINDLE_RPM = 1000.

CONTRACT = {
    'backend': 'fixed_force_velocity_v1', 'schema_version': 1,
    'observation_names': OBSERVATIONS, 'action_names': ['feed'],
    'action_mapping': ['4.5 + 4.5*a0 mm/s'],
    'target_force_n': 20., 'force_bounds_n': [20., 20.],
    'policy_controls_force': False, 'rotation_contribution': True,
    'spindle_rpm': SPINDLE_RPM, 'rpm_is_assumed_not_measured': True,
    'rotation_applied_to': 'removal estimator only; URDF dynamics unchanged',
    'velocity_model': 'rotating_disk', 'static_dwell_removal': 'p * omega * radius * dt',
    'forward_speed_bounds_mm_s': [0., 9.], 'moving_speed_bounds_mm_s': [3., 9.],
    'dwell': 'bounded periodic pause; residence otherwise follows feed',
    'second_complete_pass': False, 'local_return': False,
    'control_period_s': .008, 'policy_period_s': .08,
    'tracking_compensation': {'gain': .6, 'tau_s': .08, 'limit_mm': 4.},
    'hard_force_stop_n': 100., 'minimum_mode6_volume_ratio': .5,
    'volume_constraint_scopes': ['processing_roi', 'full_roi', 'whole_grid'],
    'entry_seconds': 2., 'entry_reference_footprint_mm': 18.,
    'source_path_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    'depth_units': 'h/K, uncalibrated',
    'deployment_compatible_with_variable_force_runtime': False,
    'training_method': 'Isaac physics evaluated bounded policy parameter search; not PPO',
}


class FixedForceVelocityActor(torch.nn.Module):
    def __init__(self, speed_profile, feedback=(0., 0., 0., 0., 0.),
                 pause_period_s=8., pause_duration_s=0., nominal_duration_s=116.4):
        super().__init__()
        profile = np.asarray(speed_profile, dtype=float)
        feedback = np.asarray(feedback, dtype=float)
        if (profile.ndim != 1 or len(profile) < 2 or not np.isfinite(profile).all()
                or np.min(profile) < 3. or np.max(profile) > 9.
                or feedback.shape != (5,) or not np.isfinite(feedback).all()
                or not 4. <= pause_period_s <= 20. or not 0. <= pause_duration_s <= 1.2):
            raise ValueError('Invalid bounded velocity/dwell parameters')
        self.register_buffer('speed_profile', torch.tensor(profile, dtype=torch.float32))
        self.register_buffer('feedback', torch.tensor(feedback, dtype=torch.float32))
        self.pause_period_s = float(pause_period_s)
        self.pause_duration_s = float(pause_duration_s)
        self.nominal_duration_s = float(nominal_duration_s)

    def forward(self, obs):
        phase = torch.clamp(obs[:, 0], 0., 1.) * (self.speed_profile.numel() - 1)
        index = torch.clamp(phase.to(torch.long), 0, self.speed_profile.numel() - 2)
        feed = self.speed_profile[index] + (phase-index)*(self.speed_profile[index+1]-self.speed_profile[index])
        p = self.feedback
        feed = torch.clamp(feed + p[0]*(obs[:, 1]-1.) + p[1]*obs[:, 7]
                           + p[2]*obs[:, 13] + p[3]*obs[:, 4] + p[4]*obs[:, 6], 3., 9.)
        elapsed = obs[:, 8] * self.nominal_duration_s
        pause = ((elapsed >= self.pause_period_s)
                 & (torch.remainder(elapsed, self.pause_period_s) < self.pause_duration_s))
        feed = torch.where(pause, torch.zeros_like(feed), feed)
        return ((feed-4.5)/4.5).unsqueeze(1)


def export_actor(path, parameters, *, name, length, protocol):
    actor = FixedForceVelocityActor(**parameters, nominal_duration_s=length/6.)
    meta = {**CONTRACT, 'candidate': name, 'parameters': parameters, 'training_protocol': protocol}
    torch.jit.save(torch.jit.script(actor), str(path),
                   _extra_files={'fixed_force_policy.json': json.dumps(meta)})
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_contract(contract):
    for key, expected in CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(f'Fixed-force contract mismatch: {key}')







from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import rotating_surface, FixedForceGeometry
