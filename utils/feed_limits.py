"""Shared digital forward/signed feed limits; no learned policy."""
import math
import numpy as np

class SignedSpeedLimiter:
    """Signed extension of the existing slew-plus-low-pass limiter."""
    def __init__(self, dt=.008):
        self.dt = dt
        self.beta = min(1., 160.*dt/(2.*16.))
        self.reset()

    def reset(self):
        self.slew = self.speed = self.acceleration = self.jerk = 0.

    def step(self, requested, enabled=True):
        if not enabled or not math.isfinite(requested):
            self.reset()
            return 0.
        self.slew += np.clip(np.clip(requested, -3., 9.)-self.slew, -16.*self.dt, 16.*self.dt)
        previous, acceleration = self.speed, self.acceleration
        self.speed += self.beta*(self.slew-self.speed)
        self.acceleration = (self.speed-previous)/self.dt
        self.jerk = (self.acceleration-acceleration)/self.dt
        return float(self.speed)


class ForwardSpeedLimiter:
    """Same 16 mm/s² and 160 mm/s³ limiter, with an explicit forward-only cap."""
    def __init__(self, maximum_speed=9., dt=.008):
        if maximum_speed not in (9., 12.) or dt != .008:
            raise ValueError('Unapproved feed limiter configuration')
        self.maximum_speed = maximum_speed
        self.dt = dt
        self.beta = min(1., 160.*dt/(2.*16.))
        self.reset()

    def reset(self):
        self.slew = self.speed = self.acceleration = self.jerk = 0.

    def step(self, requested, enabled=True):
        if not enabled or not math.isfinite(requested):
            self.reset()
            return 0.
        self.slew += np.clip(np.clip(requested, 0., self.maximum_speed)-self.slew,
                             -16.*self.dt, 16.*self.dt)
        previous, acceleration = self.speed, self.acceleration
        self.speed += self.beta*(self.slew-self.speed)
        self.acceleration = (self.speed-previous)/self.dt
        self.jerk = (self.acceleration-acceleration)/self.dt
        return float(self.speed)

