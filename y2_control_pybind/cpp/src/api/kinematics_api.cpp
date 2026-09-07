#include "y2_control_pybind/kinematics_api.hpp"
#include "y2_control_pybind/converters.hpp"
#include "Y2Kinematics/KinematicsUR10.hpp"
#include "Y2Kinematics/KinematicsUR10e.hpp"

#include <stdexcept>

namespace y2_control_pybind {

RobotKinematics::RobotKinematics(const std::string& robot_model,
                                 double dt,
                                 const std::vector<std::vector<double>>& ee2tcp)
{
    validateHTM4x4(ee2tcp, "ee2tcp");
    const YMatrix tcp = vector2dToYMatrix(ee2tcp);
    if (robot_model == "ur10") {
        kin_ = std::make_unique<KinematicsUR10>(dt, 6, tcp);
    } else if (robot_model == "ur10e") {
        kin_ = std::make_unique<KinematicsUR10e>(dt, 6, tcp);
    } else {
        throw std::invalid_argument("robot_model must be 'ur10' or 'ur10e'");
    }
}

std::vector<std::vector<double>> RobotKinematics::forward_kinematics(const std::vector<double>& q) {
    validateJointVector(q, 6, "forward_kinematics.q");
    const YMatrix T = kin_->forwardKinematics(q);
    return yMatrixToVector2d(T);
}

std::vector<std::vector<double>> RobotKinematics::calculate_jacobian(const std::vector<double>& q) {
    validateJointVector(q, 6, "calculate_jacobian.q");
    const YMatrix J = kin_->calculateJacobian(q);
    return yMatrixToVector2d(J);
}

std::vector<double> RobotKinematics::solve_ik(const std::vector<double>& q_current,
                                              const std::vector<std::vector<double>>& target_htm) {
    validateJointVector(q_current, 6, "solve_ik.q_current");
    validateHTM4x4(target_htm, "solve_ik.target_htm");

    const YMatrix target = vector2dToYMatrix(target_htm);
    return kin_->solve_IK(q_current, target);
}

void RobotKinematics::set_control_gains(double kp_pos, double kp_rot) {
    kin_->setControlGains(kp_pos, kp_rot);
}

void RobotKinematics::set_prev_q(const std::vector<double>& q_prev) {
    validateJointVector(q_prev, 6, "set_prev_q.q_prev");
    kin_->setPrevQ(q_prev);
}

void RobotKinematics::set_joint_limits(const std::vector<double>& q_min,
                                       const std::vector<double>& q_max,
                                       const std::vector<double>& qd_min,
                                       const std::vector<double>& qd_max) {
    validateJointVector(q_min, 6, "set_joint_limits.q_min");
    validateJointVector(q_max, 6, "set_joint_limits.q_max");
    validateJointVector(qd_min, 6, "set_joint_limits.qd_min");
    validateJointVector(qd_max, 6, "set_joint_limits.qd_max");

    kin_->setJointLimits(q_min, q_max, qd_min, qd_max);
}

void RobotKinematics::set_accel_limits(const std::vector<double>& a_min,
                                       const std::vector<double>& a_max) {
    validateJointVector(a_min, 6, "set_accel_limits.a_min");
    validateJointVector(a_max, 6, "set_accel_limits.a_max");

    kin_->setAccelLimits(a_min, a_max);
}

}  // namespace y2_control_pybind
