#!/usr/bin/env python3
"""Retrain the spatial policy with a lower force cap after the training audit.

The first Isaac training seed showed increased uncovered area near the entry
and endpoint. Holdout results are not used to choose this refinement.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from pathlib import Path
import numpy as np
import torch
import nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity as spatial
import nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_spatial_uniformity as trainer


def capped_action(obs, parameters):
    p = np.asarray(parameters)
    force = np.clip(20.*(1.+p[..., 0]*obs[..., 3]+p[..., 1]*obs[..., 4]+p[..., 2]), 14., 24.)
    speed = np.clip(6.*(1.+p[..., 3]*(20./force-1.)-p[..., 4]*obs[..., 13]+p[..., 5]-p[..., 6]*obs[..., 7]), 3., 9.)
    request = np.clip(p[..., 7]*obs[..., 5]-p[..., 8], -1., 1.)
    allowed = (obs[..., 14] > .5) & (obs[..., 10] > .5) & (obs[..., 11] < .5)
    return np.stack(((speed-6.)/3., (force-20.)/6., np.where(allowed, request, -1.)), axis=-1)


class CappedSpatialActor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.register_buffer('parameters_vector', torch.tensor(parameters, dtype=torch.float32))

    def forward(self, obs):
        p = self.parameters_vector
        force = torch.clamp(20.*(1.+p[0]*obs[:, 3]+p[1]*obs[:, 4]+p[2]), 14., 24.)
        speed = torch.clamp(6.*(1.+p[3]*(20./force-1.)-p[4]*obs[:, 13]+p[5]-p[6]*obs[:, 7]), 3., 9.)
        request = torch.clamp(p[7]*obs[:, 5]-p[8], -1., 1.)
        allowed = (obs[:, 14] > .5) & (obs[:, 10] > .5) & (obs[:, 11] < .5)
        request = torch.where(allowed, request, torch.full_like(request, -1.))
        return torch.stack(((speed-6.)/3., (force-20.)/6., request), dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    # Both surrogate rollout and export use exactly the same capped action.
    spatial.action_numpy = capped_action
    trainer.action_numpy = capped_action
    trainer.SpatialUniformityActor = CappedSpatialActor
    trainer.CONTRACT = {**spatial.CONTRACT, 'force_bounds_n': [14., 24.],
                        'tracking_compensation': {'gain': .6, 'tau_s': .08, 'limit_mm': 4.},
                        'refinement_basis': 'physics training seed 1501; holdouts not used',
                        'next_physics_training_seed': 1701, 'next_physics_holdout_seeds': [1801, 1802]}
    trainer.main(args)
