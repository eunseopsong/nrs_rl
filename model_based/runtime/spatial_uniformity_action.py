"""Opt-in Isaac action for the spatial policy; default Mode 5 is unchanged."""
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


class SpatialUniformityAction(AdmittanceControlAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self._num_envs_local
        self._raw_actions = torch.zeros((n, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.spatial_history = torch.zeros((n, self._action_history.shape[1], 3), device=self.device)
        self.target_force = np.full(n, 20.)
        self.force_action_center = np.full(n,20.)
        self.force_action_scale = np.full(n,6.)
        self.force_bounds = np.tile([14.,26.],(n,1))
        self.return_action = np.full(n, -1.)
        self.return_states = [LocalReturn(self.path_length_mm) for _ in range(n)]
        self.speed_limiters = [SignedSpeedLimiter(self._step_dt_local) for _ in range(n)]
        self.allow_returns = [True for _ in range(n)]

    @property
    def action_dim(self):
        return 3

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
            self.return_states[i].reset()
            self.target_force[i] = 20.
            self.return_action[i] = -1.
            self.spatial_history[i] = 0.

    def process_actions(self, actions):
        if actions.shape != self._raw_actions.shape:
            raise ValueError('Spatial policy requires feed, force, local-return actions')
        values = torch.nan_to_num(actions, nan=0., posinf=1., neginf=-1.).clamp(-1., 1.)
        # Preserve the existing speed filter and sampled action delay. The
        # temporary one-column tensor is broadcast by the parent; overwrite
        # the full diagnostic action tensor immediately afterward.
        super().process_actions(values[:, :1])
        self._raw_actions.copy_(values)
        self.spatial_history[:, 1:] = self.spatial_history[:, :-1].clone()
        self.spatial_history[:, 0] = values
        batch = torch.arange(self._num_envs_local, device=self.device)
        delayed = self.spatial_history[batch, self._action_delay]
        self._processed_actions[:, 1:] = delayed[:, 1:]
        self.return_action = delayed[:, 2].cpu().numpy().copy()

    def _control_reference(self, env_id):
        pose, force, index = super()._control_reference(env_id)
        request = self.force_action_center[env_id]+self.force_action_scale[env_id]*float(self._processed_actions[env_id, 1])
        request = np.clip(request,*self.force_bounds[env_id])
        self.target_force[env_id] += np.clip(request-self.target_force[env_id],
                                           -5.*self._step_dt_local, 5.*self._step_dt_local)
        force = force.clone()
        force[2] = torch.sign(force[2])*self.target_force[env_id]
        return pose, force, index

    def _requested_feed(self, env_id, reference_speed):
        forward = np.clip(6.+3.*float(self._filtered_action[env_id]), 3., 9.)
        request = self.return_action[env_id]
        if (not self.allow_returns[env_id] or float(self.current_abs_fz[env_id]) < 1.5
                or float(self.current_path_tracking_error_mm[env_id]) > 5.):
            request = -1.
        return self.return_states[env_id].request(float(self.path_cursor_mm[env_id]), forward, request)

    def _bounded_feed(self, speed):
        return float(np.clip(speed, -3., 9.))

    def _advance_cursor(self, env_id, distance_delta):
        return self.return_states[env_id].advance(float(self.path_cursor_mm[env_id]), distance_delta)
