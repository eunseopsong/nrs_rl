"""Small velocity-policy family with a fixed corrective feedback direction.

The four parameters control force feedforward, rate feedback, corner preview,
and nominal bias. No force command is changed. Path progress and previous
actions cannot create a time-dependent feed schedule in this actor.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import torch


PARAMETER_NAMES = ('force_gain', 'rate_gain', 'preview_gain', 'nominal_bias')
PARAMETER_LOW = (0., 0., -.6, -.08)
PARAMETER_HIGH = (1.5, .8, .6, .08)


class BoundedVelocityActor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        coefficients = torch.as_tensor(parameters, dtype=torch.float32)
        if coefficients.ndim not in (1, 2) or coefficients.shape[-1] != 4:
            raise ValueError('Expected four coefficients per policy')
        if (not torch.isfinite(coefficients).all()
                or (coefficients < torch.tensor(PARAMETER_LOW)).any()
                or (coefficients > torch.tensor(PARAMETER_HIGH)).any()):
            raise ValueError('Velocity feedback coefficients are outside their bounds')
        self.register_buffer('coefficients', coefficients.clone())
        # Zero denotes physical scaling only; there is no learned normalizer.
        self.register_buffer('obs_count', torch.tensor(0., dtype=torch.float64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        force_ratio = observations[..., 1].clamp(.5, 1.5)
        rate_error = observations[..., 8].clamp(-1., 2.)
        preview = observations[..., 9].clamp(0., 2.)
        fraction = (self.coefficients[..., 0] * (1. / force_ratio - 1.)
                    - self.coefficients[..., 1] * rate_error
                    + self.coefficients[..., 2] * preview
                    + self.coefficients[..., 3])
        # Current scheduler: feed = 6 * (1 + .5 * filtered_action).
        action = (fraction / .5).clamp(-1., 1.)
        contact = observations[..., 10] > .5
        shield = observations[..., 11] > .5
        return torch.where(contact & ~shield, action, torch.zeros_like(action)).unsqueeze(-1)
