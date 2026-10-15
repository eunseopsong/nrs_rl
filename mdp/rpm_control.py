"""Explicit virtual spindle dynamics and fixed-rate removal objective."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RPMConfig:
    nominal_rpm: float = 1000.
    log_action_scale: float = 3.
    response_tau_s: float = .016
    slew_rpm_s: float = 200000.
    physics_dt_s: float = .008
    policy_ticks: int = 2
    target_force_n: float = 20.
    pad_radius_mm: float = 15.
    rate_error_weight: float = 20.
    tail_error_weight: float = 80.
    tail_error_threshold: float = .1
    maximum_rate_loss_s: float = 100.
    invalid_contact_loss_s: float = 100.
    spatial_cv_weight: float = 100.
    incomplete_penalty: float = 100.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('RPM parameters must be finite and positive')
        if self.physics_dt_s != .008 or self.policy_ticks != 2:
            raise ValueError('Use 125 Hz physics and 62.5 Hz policy')
        if self.invalid_contact_loss_s < self.maximum_rate_loss_s:
            raise ValueError('Contact loss cannot bypass rate loss')

    @property
    def target_rate_over_k(self):
        # Continuous uniform circular pressure footprint: mean radius = 2R/3.
        return self.target_force_n * (2. * self.pad_radius_mm / 3.) * self.nominal_rpm * 2. * math.pi / 60.

    def command(self, normalized):
        if not math.isfinite(normalized) or abs(normalized) > 1.000001:
            raise ValueError('RPM action must be finite and in [-1, 1]')
        return self.nominal_rpm * math.exp(self.log_action_scale * max(-1., min(1., normalized)))

    def metadata(self):
        return {**asdict(self), 'target_rate_over_k': self.target_rate_over_k,
            'rpm_range': [self.command(-1.), self.command(1.)],
            'range_kind': 'finite numerical exploration range, not a hardware limit',
            'spindle_dynamics': 'virtual first-order response plus slew; physical USD spindle is not actuated',
            'rate_units': 'V/K per second; uncalibrated estimate',
            'target_scope': 'all measured contact including entry; pre-contact excluded'}

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class VirtualSpindle:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.rpm = self.commanded_rpm = self.config.nominal_rpm

    def step(self, action, enabled=True):
        c = self.config
        self.commanded_rpm = c.command(action) if enabled else c.nominal_rpm
        delta = -math.expm1(-c.physics_dt_s / c.response_tau_s) * (self.commanded_rpm - self.rpm)
        maximum = c.slew_rpm_s * c.physics_dt_s
        self.rpm += max(-maximum, min(maximum, delta))
        return self.rpm


class RateObjective:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.count = self.spike_count = self.invalid_count = 0
        self.sum_rate = self.sum_square_rate = self.square_error = 0.
        self.spike_square_error = self.spike_abs_error = 0.
        self.peak_abs_error = self.reward = 0.

    def update(self, rate, force, force_slew, valid=True):
        c = self.config
        if not all(math.isfinite(v) for v in (rate, force, force_slew)) or rate < 0:
            raise ValueError('Invalid measured rate')
        error = rate / c.target_rate_over_k - 1.
        spike = abs(force - c.target_force_n) >= 1. or abs(force_slew) >= 8.
        severity = 4. if spike else 1.
        raw = severity * (c.rate_error_weight * error**2
                         + c.tail_error_weight * max(abs(error) - c.tail_error_threshold, 0.)**2)
        loss = c.maximum_rate_loss_s * raw / (c.maximum_rate_loss_s + raw)
        if not valid:
            loss = c.invalid_contact_loss_s
            self.invalid_count += 1
        self.count += 1
        self.sum_rate += rate
        self.sum_square_rate += rate**2
        self.square_error += error**2
        self.peak_abs_error = max(self.peak_abs_error, abs(error))
        if spike:
            self.spike_count += 1
            self.spike_square_error += error**2
            self.spike_abs_error += abs(error) * c.physics_dt_s
        reward = -loss * c.physics_dt_s
        self.reward += reward
        return reward

    def metrics(self):
        n, ns = max(self.count, 1), max(self.spike_count, 1)
        mean = self.sum_rate / n
        variance = max(0., self.sum_square_rate / n - mean**2)
        return {'contact_samples': self.count, 'contact_seconds': self.count * self.config.physics_dt_s,
            'mean_rate_over_k': mean, 'mean_rate_ratio': mean / self.config.target_rate_over_k,
            'temporal_rate_cv': math.sqrt(variance) / mean if mean > 0. else None,
            'rate_relative_rmse': math.sqrt(self.square_error / n),
            'peak_absolute_rate_error': self.peak_abs_error,
            'spike_relative_rmse': math.sqrt(self.spike_square_error / ns),
            'spike_absolute_error_seconds': self.spike_abs_error,
            'spike_seconds': self.spike_count * self.config.physics_dt_s,
            'invalid_contact_seconds': self.invalid_count * self.config.physics_dt_s,
            'rate_reward': self.reward}
