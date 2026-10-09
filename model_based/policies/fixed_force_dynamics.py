"""Scalar residence policy and command-response surrogate for fixed-force tuning.

The surrogate includes digital feed filters, not robot/contact dynamics.
Performance acceptance always uses new full Isaac rollouts.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
import math

import numpy as np
from numba import njit
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import contract_for


class PreviewResidenceActor(torch.nn.Module):
    def __init__(self, profile, path_length_mm, lookahead_time_s=0., maximum_preview_mm=2.):
        super().__init__()
        q = np.asarray(profile, dtype=float)
        if (q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all()
                or q.min() < .5-1.e-6 or q.max() > 4.+1.e-6
                or not 0. <= lookahead_time_s <= .4 or maximum_preview_mm != 2.
                or path_length_mm <= 0.):
            raise ValueError('Invalid bounded preview/residence policy')
        self.register_buffer('residence_profile', torch.tensor(q, dtype=torch.float32))
        self.path_length_mm = float(path_length_mm)
        self.lookahead_time_s = float(lookahead_time_s)
        self.maximum_preview_mm = float(maximum_preview_mm)

    def residence(self, progress):
        phase = torch.clamp(progress, 0., 1.)*(self.residence_profile.numel()-1)
        index = torch.clamp(phase.to(torch.long), 0, self.residence_profile.numel()-2)
        return (self.residence_profile[index]
                +(phase-index)*(self.residence_profile[index+1]-self.residence_profile[index]))

    def forward(self, obs):
        phase = torch.clamp(obs[:, 0], 0., 1.)
        feed = torch.clamp(6./self.residence(phase), 1.5, 12.)
        preview = torch.clamp(feed*self.lookahead_time_s, 0., self.maximum_preview_mm)
        q = self.residence(phase+preview/self.path_length_mm)
        return (torch.clamp(6./q, 1.5, 12.)/6.-1.).unsqueeze(1)


def export_actor(path, profile, *, path_length_mm, lookahead_time_s, candidate, protocol):
    actor = PreviewResidenceActor(profile, path_length_mm, lookahead_time_s)
    contract = {**contract_for(1.5, 12.), 'candidate': candidate,
        'actor_kind': 'time_preview_residence', 'residence_profile': list(profile),
        'reference_length_mm': path_length_mm, 'lookahead_time_s': lookahead_time_s,
        'maximum_preview_mm': 2., 'preview_speed_source': 'unshifted learned profile, not noisy measured TCP speed',
        'training_protocol': protocol}
    torch.jit.save(torch.jit.script(actor), str(path),
                   _extra_files={'fixed_force_policy.json': json.dumps(contract)})
    return contract


@njit
def profile_value(profile, cursor, length):
    phase = min(1., max(0., cursor/length))*(len(profile)-1)
    left = min(int(phase), len(profile)-2)
    return profile[left]+(phase-left)*(profile[left+1]-profile[left])


@njit
def policy_feed(profile, cursor, length, lookahead_time_s):
    cursor = min(length, max(0., cursor))
    q = profile_value(profile, cursor, length)
    preview = min(2., lookahead_time_s*min(12., max(1.5, 6./q)))
    return min(12., max(1.5, 6./profile_value(profile, cursor+preview, length)))


@njit
def rollout_residence(profile, length, lookahead_time_s, delay_steps=1, samples=577):
    """Forward-only 125 Hz response with 80 ms policy decisions.

    Feed LP tau=.08, slew=18 mm/s², final acceleration=16 and jerk=160.
    Contact geometry/force response must be checked in Isaac separately.
    """
    dt = .008
    alpha = 1.-math.exp(-dt/.08)
    raw = policy_feed(profile, 0., length, lookahead_time_s)
    filtered = raw  # Policy/filter have warmed during precontact approach.
    history = np.full(4, raw)
    cursor = slew = speed = 0.
    cursors = np.zeros(42001)
    speeds = np.zeros(42001)
    speeds[0] = raw
    count = 1
    for step in range(42000):
        if step % 10 == 0:
            raw = policy_feed(profile, cursor, length, lookahead_time_s)
        history[1:] = history[:-1].copy()
        history[0] = raw
        filtered += min(.144, max(-.144, alpha*(history[delay_steps]-filtered)))
        slew += min(.128, max(-.128, filtered-slew))
        speed += .04*(slew-speed)
        cursor += speed*dt
        cursors[count] = cursor
        speeds[count] = speed
        count += 1
        if cursor >= length:
            break
    if cursor < length:
        return np.full(samples, np.nan), 336., False
    grid = np.linspace(0., length, samples)
    values = np.interp(grid, cursors[:count], speeds[:count])
    return 6./np.maximum(values, .05), (count-1)*dt, True


@njit
def replay_command_response(raw_feed, delay_steps, initial_filtered):
    """Replay captured policy outputs to audit the offline filter model."""
    n = len(raw_feed)
    filtered = initial_filtered
    slew = speed = 0.
    history = np.full(4, raw_feed[0])
    actual = np.zeros(n)
    request = np.zeros(n)
    for i in range(n):
        history[1:] = history[:-1].copy(); history[0] = raw_feed[i]
        filtered += min(.144, max(-.144, (1.-math.exp(-.1))*(history[delay_steps]-filtered)))
        request[i] = filtered
        slew += min(.128, max(-.128, filtered-slew))
        speed += .04*(slew-speed)
        actual[i] = speed
    return request, actual
