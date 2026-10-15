"""Causal, training-only removal-per-distance loss during force transients.

At fixed spindle RPM, feed primarily changes residence per path length, not
the instantaneous rotary removal rate. Use measured deposited volume and
signed TCP advance; a requested speed change earns nothing until TCP moves.
No current-speed division: stationary rotating contact must still have a cost.
"""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).with_name('config') / 'spike_dose_reward.json'


@dataclass(frozen=True)
class SpikeDoseConfig:
    sample_dt_s: float = .008
    window_s: float = .080
    force_filter_tau_s: float = .024
    reference_tau_s: float = 1.
    warmup_s: float = .240
    target_force_n: float = 20.
    warmup_force_band_n: float = 1.
    min_reference_feed_mm_s: float = .5
    force_trigger_n: float = 1.
    slew_trigger_n_s: float = 8.
    confirm_s: float = .016
    force_release_n: float = .5
    slew_release_n_s: float = 3.
    release_s: float = .400
    error_weight: float = 20.
    tail_weight: float = 80.
    tail_threshold: float = .10
    max_severity: float = 4.
    max_loss_per_s: float = 20.
    invalid_contact_loss_per_s: float = 25.
    incomplete_penalty: float = 100.
    event_action_smoothness_scale: float = .1
    loss_shape: str = 'clipped_quadratic'
    error_scale: float = .25

    def __post_init__(self):
        if self.loss_shape not in ('clipped_quadratic', 'bounded_pseudo_huber'):
            raise ValueError('Unknown spike loss shape')
        for key, value in asdict(self).items():
            if key == 'loss_shape':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.sample_dt_s != .008:
            raise ValueError('The Isaac controller samples at 125 Hz')
        if (self.window_s < self.sample_dt_s
                or not math.isclose(self.window_s / self.sample_dt_s, round(self.window_s / self.sample_dt_s))):
            raise ValueError('window_s must contain a whole number of control ticks')
        if self.force_release_n >= self.force_trigger_n or self.slew_release_n_s >= self.slew_trigger_n_s:
            raise ValueError('Release thresholds must be below trigger thresholds')
        if self.invalid_contact_loss_per_s < self.max_loss_per_s:
            raise ValueError('Losing contact must not bypass the bounded transient cost')
        if self.event_action_smoothness_scale > 1. or self.max_severity < 1.:
            raise ValueError('Invalid smoothness scale or severity cap')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        config = asdict(self)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return {'version': ('causal_spike_dose_v1' if self.loss_shape == 'clipped_quadratic'
                            else 'causal_spike_dose_v2'), 'config': config, 'config_sha256': digest,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'error': '(sum(dV/K) - frozen_dose_per_mm * sum(signed_TCP_advance_mm)) / (frozen_rate * observed_window_s)',
                'reference': 'last clean rolling-window moving-contact rate/feed; slow force baseline; frozen through event recovery',
                'scope': 'after stable moving contact; no fixed entry time or footprint exclusion',
                'units': 'uncalibrated Preston V/K; relative residual is dimensionless',
                'changes_actor_observations_or_runtime_limits': False}


class PathAdvance:
    """Project measured displacement onto the local reference tangent.

    Cursor selects direction only; scheduled cursor advance never enters the
    reward. Signed projection lets back-and-forth TCP jitter cancel in time.
    """
    def __init__(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError('A finite XYZ reference is required')
        delta = np.diff(positions, axis=0)
        length = np.linalg.norm(delta, axis=1)
        keep = length > 1.e-10
        if not np.any(keep):
            raise ValueError('Reference needs a nonzero segment')
        self.ends = length.cumsum()[keep]
        self.tangents = delta[keep] / length[keep, None]

    def project(self, displacement, cursor_mm):
        index = min(int(np.searchsorted(self.ends, cursor_mm, side='right')), len(self.ends) - 1)
        return float(np.asarray(displacement) @ self.tangents[index])


@dataclass(frozen=True)
class SpikeDoseSample:
    reward: float = 0.
    event: bool = False
    error: float = 0.
    severity: float = 0.
    invalid_contact: bool = False


class SpikeDoseReward:
    def __init__(self, config=None):
        self.config = config or SpikeDoseConfig()
        self.window_ticks = round(self.config.window_s / self.config.sample_dt_s)
        self.reset()

    def reset(self):
        self.window = deque(maxlen=self.window_ticks)
        self.filtered_force = None
        self.reference_force = self.reference_rate = self.reference_feed = None
        self.event = False
        self.pending_s = self.recovery_s = self.warmup_s = 0.
        self.event_count = 0
        self.event_seconds = self.invalid_seconds = 0.
        self.error_squared_seconds = self.absolute_error_seconds = self.peak_error = 0.
        self.total_reward = 0.

    def metrics(self):
        return {'event_count': self.event_count, 'event_seconds': self.event_seconds,
                'event_error_rms': math.sqrt(self.error_squared_seconds / self.event_seconds) if self.event_seconds else 0.,
                'event_abs_error_seconds': self.absolute_error_seconds,
                'event_peak_abs_error': self.peak_error, 'invalid_contact_seconds': self.invalid_seconds,
                'reward': self.total_reward, 'reference_established': self.reference_rate is not None}

    def update(self, force_n, volume_over_k, forward_mm, *, contact=True, shield=False):
        """Consume one measured 8 ms interval, in chronological order.

        ``volume_over_k`` is this interval's deposited volume, not a cumulative
        value. The entire rotating footprint is integrated by the caller.
        """
        if not all(math.isfinite(x) for x in (force_n, volume_over_k, forward_mm)) or force_n < 0 or volume_over_k < 0:
            raise ValueError('Finite nonnegative force/volume and finite signed advance required')
        c, dt = self.config, self.config.sample_dt_s
        if not contact or shield:
            self.window.clear()
            self.pending_s = self.recovery_s = self.warmup_s = 0.
            # Keep an established reference/event across dropouts. Never let an
            # actor erase its error by losing contact or entering a safety stop.
            penalty = c.invalid_contact_loss_per_s * dt if self.reference_rate is not None else 0.
            self.invalid_seconds += dt if self.reference_rate is not None else 0.
            self.total_reward -= penalty
            return SpikeDoseSample(-penalty, invalid_contact=True)

        old_force = self.filtered_force
        if old_force is None:
            old_force = force_n
        self.filtered_force = old_force + (-math.expm1(-dt / c.force_filter_tau_s)) * (force_n - old_force)
        slew = (self.filtered_force - old_force) / dt
        self.window.append((volume_over_k, forward_mm))
        # Always sum SIGNED displacement; neither commanded feed nor sum of
        # positive displacement can hide stationary oscillation/backtracking.
        duration = len(self.window) * dt
        volume, advance = map(sum, zip(*self.window))
        rate, feed = volume / duration, advance / duration

        if self.reference_rate is None:
            stable = (len(self.window) == self.window_ticks and feed >= c.min_reference_feed_mm_s
                      and abs(self.filtered_force - c.target_force_n) <= c.warmup_force_band_n
                      and abs(slew) <= c.slew_release_n_s and rate > 0.)
            self.warmup_s = self.warmup_s + dt if stable else 0.
            if self.warmup_s + 1.e-12 >= c.warmup_s:
                self.reference_force, self.reference_rate, self.reference_feed = self.filtered_force, rate, feed
            return SpikeDoseSample()

        deviation = abs(self.filtered_force - self.reference_force)
        suspect = deviation >= c.force_trigger_n or abs(slew) >= c.slew_trigger_n_s
        if not self.event:
            self.pending_s = self.pending_s + dt if suspect else 0.
            if self.pending_s + 1.e-12 >= c.confirm_s:
                self.event = True
                self.event_count += 1
            elif not suspect and feed >= c.min_reference_feed_mm_s and rate > 0.:
                alpha = -math.expm1(-dt / c.reference_tau_s)
                self.reference_force += alpha * (self.filtered_force - self.reference_force)
                # Follow normal CV-driven feed changes without a one-second
                # lag. Only the FORCE detector needs the slow baseline.
                self.reference_rate, self.reference_feed = rate, feed

        if not self.event:
            return SpikeDoseSample()
        # Do not adapt the reference during a sustained force step; otherwise
        # waiting, instead of compensating, would eventually remove the loss.
        dose = self.reference_rate / self.reference_feed
        error = (volume - dose * advance) / (self.reference_rate * duration)
        severity = min(c.max_severity, max(1., deviation / c.force_trigger_n, abs(slew) / c.slew_trigger_n_s))
        tail = max(abs(error) - c.tail_threshold, 0.)
        if c.loss_shape == 'bounded_pseudo_huber':
            # Smoothly bounded, strictly increasing for finite |error| > 0:
            # large-but-different errors no longer share a hard clipped penalty.
            robust = math.hypot(1., error / c.error_scale) - 1.
            raw_loss = severity * (c.error_weight * robust + c.tail_weight * tail)
            loss = c.max_loss_per_s * raw_loss / (c.max_loss_per_s + raw_loss)
        else:
            loss = min(c.max_loss_per_s, severity * (c.error_weight * error**2 + c.tail_weight * tail**2))
        reward = -dt * loss
        self.event_seconds += dt
        self.error_squared_seconds += dt * error**2
        self.absolute_error_seconds += dt * abs(error)
        self.peak_error = max(self.peak_error, abs(error))
        self.total_reward += reward
        recovered = deviation <= c.force_release_n and abs(slew) <= c.slew_release_n_s
        self.recovery_s = self.recovery_s + dt if recovered else 0.
        if self.recovery_s + 1.e-12 >= c.release_s:
            self.event = False
            self.pending_s = self.recovery_s = 0.
        return SpikeDoseSample(reward, True, error, severity)
