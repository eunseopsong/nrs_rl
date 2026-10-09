"""125 Hz process state shared conceptually with the staged Y2 scheduler.

Schema 3 has 14 inputs and a lookahead path-turn feature in slot 9. Schema 2
used immediate curvature there and is restored explicitly for old evaluations.
All filters are causal; raw evaluation
signals and force control are independent of these policy-only estimates.
"""

from __future__ import annotations

import math
import bisect
import numpy as np

OBSERVATION_VERSION = 3
OBSERVATION_SIZE = 14
OBSERVATION_FIELDS = (
    "filtered_force_error_ratio", "filtered_force_ratio", "filtered_force_derivative_100n_s",
    "filtered_speed_ratio", "tracking_error_ratio", "filtered_action",
    "applied_speed_ratio", "path_progress", "filtered_rate_error_ratio",
    "path_turn_preview", "contact", "shield", "applied_acceleration_ratio", "previous_clipped_action",
)


class PathTurnPreview:
    """A causal reference-path preview; no future measured force is available.

    Turn magnitude is 1-cos(angle), weighted linearly from zero at the preview
    boundary to full magnitude at a vertex. Distance is measured along the
    physical path projection, so tangential compliance does not shift preview
    timing by the command cursor's tracking lead.
    """
    def __init__(self, positions, arc_mm, horizon_mm):
        self.horizon = float(horizon_mm)
        if not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("turn preview horizon must be positive")
        delta = np.diff(np.asarray(positions, dtype=float), axis=0)
        direction = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        turn = np.clip(1 - np.sum(direction[:-1] * direction[1:], axis=1), 0, 2)
        useful = turn > 1e-6
        self.distance = np.asarray(arc_mm, dtype=float)[1:-1][useful]
        self.magnitude = turn[useful]

    def at(self, distance_mm):
        begin = bisect.bisect_left(self.distance, distance_mm)
        end = bisect.bisect_right(self.distance, distance_mm + self.horizon)
        if begin == end:
            return 0.
        weights = 1 - (self.distance[begin:end] - distance_mm) / self.horizon
        return float(np.max(self.magnitude[begin:end] * weights))


def tracking_feedback_action(*, filtered_speed: float, applied_speed: float,
                             acceleration: float, reference_speed: float,
                             residual_fraction: float = 0.20, gain: float = 0.5,
                             signal_tau: float = 0.08) -> float:
    """Causal diagnostic policy for the speed-tracking part of the rate error.

    The first-order command lag correction avoids interpreting the process
    observation filter's delay as a physical tracking disturbance. This is an
    explicit scripted comparator, not a trained policy or a new reward metric.
    The existing action filter and native speed limiter still apply afterward.
    """
    error = applied_speed - signal_tau * acceleration - filtered_speed
    scale = max(reference_speed * residual_fraction, 1.0e-6)
    return min(1.0, max(-1.0, gain * error / scale))


class ProcessState:
    def __init__(self, dt: float = 0.008, tau: float = 0.08):
        if dt <= 0 or tau <= 0:
            raise ValueError("dt and tau must be positive")
        self.dt = dt
        self.alpha = -math.expm1(-dt / tau)
        self.reset()

    def reset(self):
        self.valid = False
        self.force = self.speed = self.rate = self.force_derivative = 0.0

    def update(self, force: float, speed: float, rate: float):
        if not all(math.isfinite(x) for x in (force, speed, rate)):
            self.reset()
            return
        if not self.valid:
            self.force, self.speed, self.rate = force, speed, rate
            self.force_derivative = 0.0
            self.valid = True
            return
        previous_force = self.force
        self.force += self.alpha * (force - self.force)
        self.speed += self.alpha * (speed - self.speed)
        self.rate += self.alpha * (rate - self.rate)
        self.force_derivative = (self.force - previous_force) / self.dt

    def reference_speed(self, nominal: float, target_rate: float, contact_force: float,
                        compensation: bool = True) -> float:
        # A process prior supplies the inverse-force relation. PPO learns only
        # a small correction for lag/contact dynamics, starting at action zero.
        # No inverse-force gain during contact acquisition or a dropout.
        if not compensation or not self.valid or self.force < contact_force:
            return nominal
        return min(1.25 * nominal, max(0.75 * nominal, target_rate / self.force))

    def observation(self, *, target_force, max_speed, tracking_error, tracking_stop,
                    filtered_action, applied_speed, progress, target_rate, curvature,
                    contact, shield, acceleration, max_acceleration, clipped_action):
        force_scale = max(target_force, 1.0)
        return (
            (self.force - target_force) / force_scale, self.force / force_scale,
            self.force_derivative / 100.0, self.speed / max_speed,
            tracking_error / tracking_stop, filtered_action, applied_speed / max_speed,
            progress, (self.rate - target_rate) / target_rate, curvature,
            float(contact), float(shield), acceleration / max_acceleration, clipped_action,
        )
