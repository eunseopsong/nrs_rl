"""Expanded scalar velocity/residence policy; target force and RPM stay fixed."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
import math
from pathlib import Path

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT as PREVIOUS_CONTRACT, check_contract as check_previous


def contract_for(minimum_speed, maximum_speed):
    if minimum_speed not in (.75, 1.5, 3.) or maximum_speed not in (9., 12.):
        raise ValueError('Unapproved experimental feed bounds')
    half = maximum_speed/2.
    return {**PREVIOUS_CONTRACT, 'backend': 'fixed_force_velocity_v2',
        'action_mapping': [f'{half:g} + {half:g}*a0 mm/s'],
        'speed_mapping': {'center_mm_s': half, 'scale_mm_s': half},
        'forward_speed_bounds_mm_s': [0., maximum_speed],
        'moving_speed_bounds_mm_s': [minimum_speed, maximum_speed],
        'profile_interpolation': 'linear inverse speed (residence per distance)',
        'dwell': 'positive bounded residence profile; no periodic pauses or reversals',
        'feed_action_slew_mm_s2': 18., 'speed_acceleration_limit_mm_s2': 16.,
        'speed_jerk_limit_mm_s3': 160.,
        'training_method': 'constrained residence optimization with fixed-force trace replay and Isaac selection; not PPO'}


def check_contract(contract):
    if contract.get('backend') == PREVIOUS_CONTRACT['backend']:
        check_previous(contract)
        return 3., 9.
    minimum, maximum = contract.get('moving_speed_bounds_mm_s', (None, None))
    expected = contract_for(minimum, maximum)
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Fixed-force refinement contract mismatch: {key}')
    return minimum, maximum


class ResidenceActor(torch.nn.Module):
    def __init__(self, residence_profile, *, minimum_speed=1.5, maximum_speed=12.,
                 phase_lead_mm=0., path_length_mm=698.):
        super().__init__()
        contract_for(minimum_speed, maximum_speed)
        profile = np.asarray(residence_profile, dtype=float)
        if (profile.ndim != 1 or len(profile) < 2 or not np.isfinite(profile).all()
                or profile.min() < 6./maximum_speed-1.e-6
                or profile.max() > 6./minimum_speed+1.e-6
                or not 0. <= phase_lead_mm <= 2. or path_length_mm <= 0.):
            raise ValueError('Invalid residence profile')
        self.register_buffer('residence_profile', torch.tensor(profile, dtype=torch.float32))
        self.minimum_speed = float(minimum_speed)
        self.maximum_speed = float(maximum_speed)
        self.phase_lead_mm = float(phase_lead_mm)
        self.path_length_mm = float(path_length_mm)

    def forward(self, obs):
        phase = torch.clamp(obs[:, 0]+self.phase_lead_mm/self.path_length_mm, 0., 1.)
        phase *= self.residence_profile.numel()-1
        index = torch.clamp(phase.to(torch.long), 0, self.residence_profile.numel()-2)
        residence = (self.residence_profile[index]
                     +(phase-index)*(self.residence_profile[index+1]-self.residence_profile[index]))
        feed = torch.clamp(6./residence, self.minimum_speed, self.maximum_speed)
        return (feed/(self.maximum_speed/2.)-1.).unsqueeze(1)


def export_actor(path, profile, *, minimum_speed, maximum_speed, phase_lead_mm,
                 path_length_mm, candidate, protocol):
    actor = ResidenceActor(profile, minimum_speed=minimum_speed, maximum_speed=maximum_speed,
                           phase_lead_mm=phase_lead_mm, path_length_mm=path_length_mm)
    contract = {**contract_for(minimum_speed, maximum_speed), 'candidate': candidate,
                'residence_profile': list(profile), 'phase_lead_mm': phase_lead_mm,
                'reference_length_mm': path_length_mm, 'training_protocol': protocol}
    torch.jit.save(torch.jit.script(actor), str(path),
                   _extra_files={'fixed_force_policy.json': json.dumps(contract)})
    return contract


from nrs_rl.tasks.manager_based.nrs_rl.utils.feed_limits import ForwardSpeedLimiter
