"""Allow the independently tagged PPO actor through the fixed-force Isaac action."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.model_based.runtime.fixed_force_refinement_action import FixedForceRefinementAction
from nrs_rl.tasks.model_based.policies.fixed_force_refinement import ForwardSpeedLimiter
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import check_ppo_or_baseline


class FixedForcePPOAction(FixedForceRefinementAction):
    def configure_velocity(self, env_id, contract):
        _, maximum = check_ppo_or_baseline(contract)
        self.half_range[env_id] = maximum/2.
        self.speed_limiters[env_id] = ForwardSpeedLimiter(maximum, self._step_dt_local)
