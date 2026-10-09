"""Bounded outer tracking feedback, applied only in the TCP tangent plane.

This experimental controller wrapper is separate from the running validated
runtime. Gain zero is exactly the original controller. No normal-force target,
material metric, robot model or servo stiffness is changed.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math
import json
import numpy as np


class TangentTrackingCompensation:
    def __init__(self, core, gain, *, dt=.008, tau=.08, limit_mm=4.):
        if (not all(math.isfinite(v) for v in (gain, dt, tau, limit_mm))
                or not 0. <= gain <= 1. or dt != .008 or tau != .08 or limit_mm != 4.):
            raise ValueError('Tracking compensation outside the frozen experimental contract')
        self.core = core
        self.gain = float(gain)
        self.alpha = -math.expm1(-dt/tau)
        self.limit_mm = limit_mm
        self.correction = np.zeros(3)

    def reset(self, pose):
        self.correction.fill(0.)
        self.core.reset(pose)

    def step(self, measured, reference, force, wrench, rotation):
        if self.gain == 0.:
            return self.core.step(measured, reference, force, wrench, rotation)
        pose = np.asarray(measured, dtype=float)
        target = np.asarray(reference, dtype=float).copy()
        matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
        normal = matrix[:, 2]
        if (not np.isfinite(pose).all() or not np.isfinite(target).all()
                or not np.isfinite(matrix).all() or not np.isclose(normal@normal, 1., atol=1e-5)):
            raise ValueError('Invalid tracking-compensation pose or TCP normal')
        error = target[:3]-pose[:3]
        tangent = error-normal*(error@normal)
        requested = self.gain*tangent if abs(force[2]) > .1 else np.zeros(3)
        norm = np.linalg.norm(requested)
        requested *= min(1., self.limit_mm/max(norm, 1e-12))
        self.correction += self.alpha*(requested-self.correction)
        self.correction -= normal*(self.correction@normal)
        norm = np.linalg.norm(self.correction)
        self.correction *= min(1., self.limit_mm/max(norm, 1e-12))
        target[:3] += self.correction
        return self.core.step(measured, target.tolist(), force, wrench, rotation)


class TrackingPolicyAgent:
    """Explicit experimental loader; the regular deployment worker rejects this backend."""
    def __init__(self, checkpoint, contract):
        import torch
        if contract['policy_backend'] != 'torchscript_tcp_tracking':
            raise ValueError('Expected the tracking-compensation experiment backend')
        extra = {'velocity_policy.json': ''}
        self.actor = torch.jit.load(str(checkpoint), map_location='cpu', _extra_files=extra).eval()
        if json.loads(extra['velocity_policy.json']) != contract:
            raise ValueError('Embedded tracking controller contract differs from its sidecar')
        self.velocity_contract = contract

    def act(self, observations, timestep=0, timesteps=0):
        mean = self.actor(observations)
        return mean, None, {'mean_actions': mean}
