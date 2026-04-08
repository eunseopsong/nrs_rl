#include "y2_control_pybind/kinematics_api.hpp"
#include "y2_control_pybind/converters.hpp"

namespace y2_control_pybind {

UR10eKinematics::UR10eKinematics(double dt,
                                 const std::vector<std::vector<double>>& ee2tcp)
    : kin_(dt, 6, vector2dToYMatrix(ee2tcp)) {
    validateHTM4x4(ee2tcp, "ee2tcp");
}

std::vector<std::vector<double>> UR10eKinematics::forward_kinematics(const std::vector<double>& q) {
    validateJointVector(q, 6, "forward_kinematics.q");
    const YMatrix T = kin_.forwardKinematics(q);
    return yMatrixToVector2d(T);
}

std::vector<std::vector<double>> UR10eKinematics::calculate_jacobian(const std::vector<double>& q) {
    validateJointVector(q, 6, "calculate_jacobian.q");
    const YMatrix J = kin_.calculateJacobian(q);
    return yMatrixToVector2d(J);
}

std::vector<double> UR10eKinematics::solve_ik(const std::vector<double>& q_current,
                                              const std::vector<std::vector<double>>& target_htm) {
    validateJointVector(q_current, 6, "solve_ik.q_current");
    validateHTM4x4(target_htm, "solve_ik.target_htm");

    const YMatrix target = vector2dToYMatrix(target_htm);
    return kin_.solve_IK(q_current, target);
}

void UR10eKinematics::set_control_gains(double kp_pos, double kp_rot) {
    kin_.setControlGains(kp_pos, kp_rot);
}

void UR10eKinematics::set_prev_q(const std::vector<double>& q_prev) {
    validateJointVector(q_prev, 6, "set_prev_q.q_prev");
    kin_.setPrevQ(q_prev);
}

void UR10eKinematics::set_joint_limits(const std::vector<double>& q_min,
                                       const std::vector<double>& q_max,
                                       const std::vector<double>& qd_min,
                                       const std::vector<double>& qd_max) {
    validateJointVector(q_min, 6, "set_joint_limits.q_min");
    validateJointVector(q_max, 6, "set_joint_limits.q_max");
    validateJointVector(qd_min, 6, "set_joint_limits.qd_min");
    validateJointVector(qd_max, 6, "set_joint_limits.qd_max");

    kin_.setJointLimits(q_min, q_max, qd_min, qd_max);
}

void UR10eKinematics::set_accel_limits(const std::vector<double>& a_min,
                                       const std::vector<double>& a_max) {
    validateJointVector(a_min, 6, "set_accel_limits.a_min");
    validateJointVector(a_max, 6, "set_accel_limits.a_max");

    kin_.setAccelLimits(a_min, a_max);
}

}  // namespace y2_control_pybind