#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Copy this header unchanged into Y2RobMotion/include/Y2RobMotion at deployment.
// Units: mm/s, mm/s^2, mm/s^3, seconds. No allocation in the 125 Hz step.
class PpoSpeedLimiter
{
public:
    PpoSpeedLimiter(double dt = 0.008, double max_speed = 12.0,
                    double max_acceleration = 16.0, double max_jerk = 160.0)
        : dt_(dt), max_speed_(max_speed), max_acceleration_(max_acceleration)
    {
        for (double x : {dt, max_speed, max_acceleration, max_jerk}) {
            if (!std::isfinite(x) || x <= 0.0) {
                throw std::invalid_argument("PpoSpeedLimiter parameters must be finite and positive");
            }
        }
        // v' is a convex combination of bounded slew-stage accelerations.
        // |a| <= A and |a[k]-a[k-1]|/dt <= 2*beta*A/dt <= J.
        beta_ = std::min(1.0, max_jerk * dt / (2.0 * max_acceleration));
    }

    void reset()
    {
        slew_speed_ = speed_ = acceleration_ = jerk_ = 0.0;
    }

    double step(double requested_speed, bool enabled = true)
    {
        if (!enabled || !std::isfinite(requested_speed)) {
            // Safety stops override comfort limits. Reset avoids a release kick.
            reset();
            return 0.0;
        }
        const double target = std::clamp(requested_speed, 0.0, max_speed_);
        slew_speed_ += std::clamp(target - slew_speed_,
                                  -max_acceleration_ * dt_, max_acceleration_ * dt_);
        const double previous_speed = speed_;
        const double previous_acceleration = acceleration_;
        speed_ += beta_ * (slew_speed_ - speed_);
        acceleration_ = (speed_ - previous_speed) / dt_;
        jerk_ = (acceleration_ - previous_acceleration) / dt_;
        return speed_;
    }

    double speed() const { return speed_; }
    double acceleration() const { return acceleration_; }
    double jerk() const { return jerk_; }

private:
    double dt_, max_speed_, max_acceleration_, beta_;
    double slew_speed_ = 0.0, speed_ = 0.0, acceleration_ = 0.0, jerk_ = 0.0;
};
