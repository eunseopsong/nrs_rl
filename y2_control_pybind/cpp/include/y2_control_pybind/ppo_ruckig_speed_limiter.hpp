#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <ruckig/ruckig.hpp>

// Velocity-interface OTG, in mm/s, mm/s^2, mm/s^3 and seconds. Limits are
// identical to PpoSpeedLimiter; no extra fixed 0.2 s low-pass stage is added.
// Copy this header unchanged to Y2RobMotion when deploying a Ruckig policy.
class PpoRuckigSpeedLimiter
{
public:
    PpoRuckigSpeedLimiter(double dt = 0.008, double max_speed = 12.0,
                         double max_acceleration = 16.0, double max_jerk = 160.0)
        : generator_(dt), dt_(dt), max_speed_(max_speed)
    {
        for(double value : {dt, max_speed, max_acceleration, max_jerk}) {
            if(!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument("Speed limiter parameters must be finite and positive");
        }
        input_.control_interface = ruckig::ControlInterface::Velocity;
        input_.target_position = {0.0};
        input_.target_velocity = {0.0};
        input_.target_acceleration = {0.0};
        input_.max_velocity = {max_speed};
        input_.max_acceleration = {max_acceleration};
        input_.max_jerk = {max_jerk};
        reset();
    }

    void reset()
    {
        input_.current_position = {0.0};
        input_.current_velocity = {0.0};
        input_.current_acceleration = {0.0};
        speed_ = acceleration_ = jerk_ = 0.0;
        generator_.reset();
    }

    double step(double requested_speed, bool enabled = true)
    {
        if(!enabled || !std::isfinite(requested_speed)) {
            reset();
            return 0.0;
        }
        input_.target_velocity[0] = std::clamp(requested_speed, 0.0, max_speed_);
        const auto result = generator_.update(input_, output_);
        if(static_cast<int>(result) < 0) {
            reset();
            throw std::runtime_error("Ruckig velocity update failed");
        }
        output_.pass_to_input(input_);
        const double next_speed = input_.current_velocity[0];
        if(!std::isfinite(next_speed) || next_speed < -1.0e-8 || next_speed > max_speed_ + 1.0e-8) {
            reset();
            throw std::runtime_error("Ruckig velocity exceeded command bounds");
        }
        jerk_ = (input_.current_acceleration[0] - acceleration_) / dt_;
        acceleration_ = input_.current_acceleration[0];
        // Only remove roundoff at the hard speed bounds.
        speed_ = std::clamp(next_speed, 0.0, max_speed_);
        return speed_;
    }

    double speed() const { return speed_; }
    double acceleration() const { return acceleration_; }
    double jerk() const { return jerk_; }

private:
    ruckig::Ruckig<1> generator_;
    ruckig::InputParameter<1> input_;
    ruckig::OutputParameter<1> output_;
    double dt_, max_speed_;
    double speed_ = 0.0, acceleration_ = 0.0, jerk_ = 0.0;
};
