"""Isaac scalar feed action with per-checkpoint scaling and fixed 20 N target."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math
import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.runtime.fixed_force_velocity_action import FixedForceVelocityAction
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import ForwardSpeedLimiter, check_contract


class FixedForceRefinementAction(FixedForceVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.half_range = np.full(self._num_envs_local, 4.5)

    def configure_velocity(self, env_id, contract):
        _, maximum = check_contract(contract)
        self.half_range[env_id] = maximum/2.
        self.speed_limiters[env_id] = ForwardSpeedLimiter(maximum, self._step_dt_local)

    def process_actions(self, actions):
        if actions.shape != (self._num_envs_local, 1):
            raise ValueError('Only scalar feed is allowed; force and reverse actions forbidden')
        if not torch.isfinite(actions).all() or torch.any(actions.abs() > 1.000001):
            raise ValueError('Invalid scalar feed action')
        # Preserve the old filter exactly at 9 mm/s. At 12 mm/s, rescale the
        # normalized slew so its physical limit remains the same 18 mm/s².
        self._control_tick_pending = True
        self._raw_actions.copy_(actions)
        clipped = actions[:, 0].clamp(-1., 1.)
        self.action_delta.copy_(clipped-self._previous_clipped_action)
        self._previous_clipped_action.copy_(clipped)
        self._action_history[:, 1:] = self._action_history[:, :-1].clone()
        self._action_history[:, 0] = clipped
        delayed = self._action_history.gather(1, self._action_delay[:, None]).squeeze(1)
        alpha = 1.-math.exp(-self._step_dt_local/max(self.int_cfg.action_filter_tau_s, 1.e-6))
        target = self._filtered_action+alpha*(delayed-self._filtered_action)
        maximum_delta = torch.as_tensor(18.*self._step_dt_local/self.half_range, device=self.device)
        increment = torch.maximum(torch.minimum(target-self._filtered_action, maximum_delta), -maximum_delta)
        self._filtered_action += increment.to(self._filtered_action.dtype)
        self._processed_actions[:, 0] = self._filtered_action

    def _requested_feed(self, env_id, reference_speed):
        half = self.half_range[env_id]
        return float(np.clip(half+half*float(self._filtered_action[env_id]), 0., 2.*half))

    def _bounded_feed(self, speed):
        return float(np.clip(speed, 0., 12.))
