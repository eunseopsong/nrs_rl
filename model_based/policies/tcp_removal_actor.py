"""Bounded causal feed policy for RPM-free F*TCP-distance removal.

Episode rewards fit six coefficients in the simulator. The 14 observations,
one action and native 3--9 mm/s smoothing contract remain unchanged. In
particular, the policy cannot raise the force target to obtain more removal.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import torch

PARAMETER_NAMES = ('inverse_force_gain', 'rate_gain', 'corner_slowdown',
                   'tracking_slowdown', 'nominal_bias', 'force_slope_damping')
LOW = (0., 0., 0., 0., -.06, 0.)
HIGH = (1.5, .5, .65, .5, .18, .3)


class TcpRemovalActor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        coefficients = torch.as_tensor(parameters, dtype=torch.float32)
        if (coefficients.ndim not in (1, 2) or coefficients.shape[-1] != 6
                or not torch.isfinite(coefficients).all()
                or (coefficients < torch.tensor(LOW)).any()
                or (coefficients > torch.tensor(HIGH)).any()):
            raise ValueError('Invalid TCP-removal policy parameters')
        self.register_buffer('coefficients', coefficients.clone())
        self.register_buffer('obs_count', torch.tensor(0., dtype=torch.float64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        force = observations[..., 1].clamp(.5, 1.5)
        rate = observations[..., 8].clamp(-1., 2.)
        turn = observations[..., 9].clamp(0., 1.)
        tracking = (observations[..., 4]-.1).clamp(0., 1.)
        rising_force = observations[..., 2].clamp(0., 1.)
        c = self.coefficients
        request = (c[..., 0]*(1./force-1.) - c[..., 1]*rate
                   - c[..., 2]*turn - c[..., 3]*tracking
                   + c[..., 4] - c[..., 5]*rising_force)
        action = (request/.5).clamp(-1., 1.)
        valid = (observations[..., 10]>.5) & (observations[..., 11]<.5)
        return torch.where(valid, action, torch.zeros_like(action)).unsqueeze(-1)


class SignedTcpRemovalActor(TcpRemovalActor):
    """Test both signs of force/feed coupling with the same bounded action.

    Used only by the separate force-stability study. Original positive-gain
    policy validation and all already exported checkpoints are unchanged.
    """
    def __init__(self, parameters):
        values = torch.as_tensor(parameters, dtype=torch.float32)
        positive = values.clone()
        if positive.ndim not in (1, 2) or positive.shape[-1] != 6:
            raise ValueError('Invalid signed TCP-removal policy parameters')
        positive[..., 0] = positive[..., 0].abs()
        super().__init__(positive)
        self.coefficients.copy_(values)
