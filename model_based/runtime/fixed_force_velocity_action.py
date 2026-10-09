"""Fixed 20 N force target with a scalar feed/dwell action, for Isaac only."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.action import AdmittanceControlAction
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import SignedSpeedLimiter, LocalReturn


class FixedForceVelocityAction(AdmittanceControlAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self._num_envs_local
        self.target_force = np.full(n, 20.)
        self.speed_limiters = [SignedSpeedLimiter(self._step_dt_local) for _ in range(n)]
        # Retain the observation bookkeeping without any reverse requests.
        self.return_states = [LocalReturn(self.path_length_mm) for _ in range(n)]

    @property
    def action_dim(self):
        return 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if not hasattr(self, 'return_states'):
            return
        if env_ids is None:
            ids = range(self._num_envs_local)
        elif isinstance(env_ids, slice):
            ids = range(self._num_envs_local)[env_ids]
        else:
            ids = torch.as_tensor(env_ids).cpu().tolist()
        for i in ids:
            self.target_force[i] = 20.
            self.return_states[i].reset()

    def process_actions(self, actions):
        if actions.shape != (self._num_envs_local, 1):
            raise ValueError('Only one feed action is accepted; force/return actions forbidden')
        if not torch.isfinite(actions).all() or torch.any(torch.abs(actions) > 1.000001):
            raise ValueError('Invalid bounded scalar velocity action')
        super().process_actions(actions)

    def _control_reference(self, env_id):
        pose, force, index = super()._control_reference(env_id)
        force = torch.zeros_like(force)
        force[2] = 20.  # Fixed sign/magnitude for this +Z 20 N reference.
        self.target_force[env_id] = 20.
        return pose, force, index

    def _requested_feed(self, env_id, reference_speed):
        return float(np.clip(4.5 + 4.5*float(self._filtered_action[env_id]), 0., 9.))

    def _bounded_feed(self, speed):
        return float(np.clip(speed, 0., 9.))

    def _advance_cursor(self, env_id, distance_delta):
        return self.return_states[env_id].advance(float(self.path_cursor_mm[env_id]), max(0., distance_delta))
