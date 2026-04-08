#pragma once

#include "Y2Kinematics/KinematicsUR10e.hpp"

#include <vector>

namespace y2_control_pybind {

class UR10eKinematics {
public:
    UR10eKinematics(
        double dt = 0.01,
        const std::vector<std::vector<double>>& ee2tcp = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
        }
    );

    std::vector<std::vector<double>> forward_kinematics(const std::vector<double>& q);
    std::vector<std::vector<double>> calculate_jacobian(const std::vector<double>& q);
    std::vector<double> solve_ik(const std::vector<double>& q_current,
                                 const std::vector<std::vector<double>>& target_htm);

    void set_control_gains(double kp_pos, double kp_rot);
    void set_prev_q(const std::vector<double>& q_prev);

    void set_joint_limits(const std::vector<double>& q_min,
                          const std::vector<double>& q_max,
                          const std::vector<double>& qd_min,
                          const std::vector<double>& qd_max);

    void set_accel_limits(const std::vector<double>& a_min,
                          const std::vector<double>& a_max);

private:
    KinematicsUR10e kin_;
};

}  // namespace y2_control_pybind