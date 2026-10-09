"""125 Hz feed policy around a smooth, explicitly modeled dwell profile.

Constant-RPM, rotation-dominated Preston removal scales with force and dwell.
Positive force correction therefore increases feed. Parameter search learns
profile strength, force correction, negative speed feedback, and a small bias.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
from pathlib import Path
import torch


PARAMETER_NAMES = ('profile_strength', 'force_gain', 'speed_gain', 'nominal_bias')
LOW = (0., 0., 0., -.05)
HIGH = (1., 1.5, .8, .05)


class PrestonDwellActor(torch.nn.Module):
    def __init__(self, profile, parameters, execution_path_length_mm=None):
        super().__init__()
        knots = torch.tensor(profile['cursor_knots_mm'], dtype=torch.float32)
        dwell = torch.tensor(profile['dwell_multipliers'], dtype=torch.float32)
        coefficients = torch.as_tensor(parameters, dtype=torch.float32)
        if (len(knots) < 2 or knots.shape != dwell.shape or knots[0] != 0
                or not torch.isfinite(knots).all() or not torch.isfinite(dwell).all()
                or not torch.all(knots.diff() > 0) or (dwell < 2/3-1e-6).any() or (dwell > 2+1e-6).any()):
            raise ValueError('Expected a finite 3--9 mm/s dwell profile with increasing cursor knots')
        if (coefficients.ndim not in (1, 2) or coefficients.shape[-1] != 4
                or not torch.isfinite(coefficients).all()
                or (coefficients < torch.tensor(LOW)).any() or (coefficients > torch.tensor(HIGH)).any()):
            raise ValueError('Invalid dwell-policy coefficients')
        length = float(knots[-1]) if execution_path_length_mm is None else float(execution_path_length_mm)
        if not 0 < length <= float(knots[-1]) + .1:
            raise ValueError('Execution path must be the full planned path or its prefix')
        self.execution_length = length
        self.register_buffer('knots', knots)
        self.register_buffer('dwell', dwell)
        self.register_buffer('coefficients', coefficients.clone())
        self.register_buffer('obs_count', torch.tensor(0., dtype=torch.float64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cursor = observations[..., 7].clamp(0, 1) * self.execution_length
        right = torch.searchsorted(self.knots, cursor.contiguous()).clamp(1, len(self.knots)-1)
        fraction = ((cursor-self.knots[right-1]) / (self.knots[right]-self.knots[right-1])).clamp(0, 1)
        dwell = self.dwell[right-1] + fraction*(self.dwell[right]-self.dwell[right-1])
        nominal = 1. / (1. + self.coefficients[..., 0]*(dwell-1.))
        force_ratio = observations[..., 1].clamp(.5, 1.5)
        desired = nominal*(1. + self.coefficients[..., 1]*(force_ratio-1.))
        speed_ratio = (2*observations[..., 3]).clamp(0, 3)
        request_ratio = desired + self.coefficients[..., 2]*(desired-speed_ratio) + self.coefficients[..., 3]
        action = ((request_ratio-1.)/.5).clamp(-1, 1)
        valid = (observations[..., 10] > .5) & (observations[..., 11] < .5)
        return torch.where(valid, action, torch.zeros_like(action)).unsqueeze(-1)


def export_dwell_actor(profile, parameters, checkpoint, contract):
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if (contract.get('policy_backend') != 'torchscript_preston_depth'
            or contract.get('force_rate_compensation') is not False
            or contract.get('nominal_speed_mm_s') != 6.
            or contract.get('residual_speed_fraction') != .5):
        raise ValueError('The dwell actor requires its explicit backend and the direct 6 mm/s scheduler')
    actor = PrestonDwellActor(profile, parameters).eval()
    scripted = torch.jit.script(actor)
    torch.jit.save(scripted, str(checkpoint),
                   _extra_files={'velocity_policy.json': json.dumps(contract, sort_keys=True)})
    return scripted
