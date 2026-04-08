#include "Y2Kinematics/Kinematics.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace {

std::vector<double> clampToBounds(const std::vector<double>& values,
                                  const std::vector<double>& lb,
                                  const std::vector<double>& ub) {
    std::vector<double> clamped = values;
    for (size_t i = 0; i < clamped.size(); ++i) {
        clamped[i] = std::max(lb[i], std::min(ub[i], clamped[i]));
    }
    return clamped;
}

YMatrix vectorToColumnMatrix(const std::vector<double>& values) {
    YMatrix column(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        column[i][0] = values[i];
    }
    return column;
}

}

// Constructor
Kinematics::Kinematics(double SamplingTime_, size_t numOfAxis_, const YMatrix& EE2TCP_)
: dt(SamplingTime_), numOfAxis(numOfAxis_), EE2TCP(EE2TCP_) {

    // tolerance from precision
    QP_tolerance = std::pow(10.0, -static_cast<double>(QP_precision));

    // Joint limits default
    q_min.assign(numOfAxis, -2.0 * M_PI);
    q_max.assign(numOfAxis,  2.0 * M_PI);

    qd_min.assign(numOfAxis, (q_min[0] / 2.0) / dt);
    qd_max.assign(numOfAxis, (q_max[0] / 2.0) / dt);

    // Accel limits default: unbounded
    a_min.assign(numOfAxis, -std::numeric_limits<double>::infinity());
    a_max.assign(numOfAxis,  std::numeric_limits<double>::infinity());

    q_prev.resize(numOfAxis, 0.0);
    has_prev = false;
}

// Enhanced QP-based IK solver
std::vector<double> Kinematics::solve_IK(const std::vector<double>& q_current,
                                         const YMatrix& target_HTM_) {

    if (q_current.size() != numOfAxis) {
        throw std::invalid_argument("solve_IK: q_current size mismatch");
    }

    // 0) Align target to TCP frame
    YMatrix target_HTM = target_HTM_ * EE2TCP.inverse();

    // 1) Current EE pose -> TCP
    YMatrix current_HTM = forwardKinematics(q_current);
    current_HTM = current_HTM * EE2TCP.inverse();

    // numeric rounding (optional)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            current_HTM[i][j] = roundToNthDecimal(current_HTM[i][j], QP_precision);
            target_HTM[i][j]  = roundToNthDecimal(target_HTM[i][j],  QP_precision);
        }
    }

    // 2) Pose error -> Delta_x_des (6)
    std::vector<double> e_pos(3);
    for (int i = 0; i < 3; ++i) e_pos[i] = target_HTM[i][3] - current_HTM[i][3];

    YMatrix R_target  = target_HTM.extract(0, 0, 3, 3);
    YMatrix R_current = current_HTM.extract(0, 0, 3, 3);
    YMatrix R_err = R_target * R_current.transpose();

    std::vector<double> e_rot(3);
    e_rot[0] = 0.5 * (R_err[2][1] - R_err[1][2]);
    e_rot[1] = 0.5 * (R_err[0][2] - R_err[2][0]);
    e_rot[2] = 0.5 * (R_err[1][0] - R_err[0][1]);

    std::vector<double> Delta_x_des(6);
    for (int i = 0; i < 3; ++i) {
        Delta_x_des[i]   = Kp_pos * e_pos[i];
        Delta_x_des[i+3] = Kp_rot * e_rot[i];
    }

    // 3) Jacobian
    YMatrix J = calculateJacobian(q_current);
    if (J.rows() != 6 || J.cols() != numOfAxis) {
        std::cerr << "[Kinematics] Jacobian size error: got "
                  << J.rows() << "x" << J.cols()
                  << ", expected 6x" << numOfAxis << "\n";
        return q_current;
    }

    // 4) Bounds on q_next (angle/vel/acc intersection)
    std::vector<double> lb(numOfAxis), ub(numOfAxis);

    for (size_t i = 0; i < numOfAxis; ++i) {
        const double lb_angle = q_min[i];
        const double ub_angle = q_max[i];

        const double lb_vel = q_current[i] + dt * qd_min[i];
        const double ub_vel = q_current[i] + dt * qd_max[i];

        double lb_acc = -std::numeric_limits<double>::infinity();
        double ub_acc =  std::numeric_limits<double>::infinity();
        if (has_prev) {
            lb_acc = dt*dt * a_min[i] + 2.0 * q_current[i] - q_prev[i];
            ub_acc = dt*dt * a_max[i] + 2.0 * q_current[i] - q_prev[i];
        }

        lb[i] = std::max({lb_angle, lb_vel, lb_acc});
        ub[i] = std::min({ub_angle, ub_vel, ub_acc});

        if (lb[i] > ub[i]) {
            // infeasible: collapse to midpoint (minimal disturbance)
            const double mid = 0.5 * (lb[i] + ub[i]);
            lb[i] = mid - 1e-9;
            ub[i] = mid + 1e-9;
        }
    }

    // 5) Solve IK with selected backend
#if Y2_IK_SOLVER_MODE == Y2_IK_SOLVER_QP
    YMatrix I   = YMatrix::identity(numOfAxis);
    YMatrix JtJ = J.transpose() * J;

    // H = 2*omega_p*I + 2*alpha*(J'J + lambda*I)^{-1}
    YMatrix H1 = I * (2.0 * omega_p);

    YMatrix J_metric     = JtJ + I * lambda;
    YMatrix J_metric_inv = J_metric.inverse();
    YMatrix H2 = J_metric_inv * (2.0 * alpha);

    YMatrix H = H1 + H2;

    // f = -2*omega_p*q_current
    std::vector<double> f(numOfAxis, 0.0);
    for (size_t i = 0; i < numOfAxis; ++i) {
        f[i] = -2.0 * omega_p * q_current[i];
    }

    // Equality constraint: J*q_next = Delta_x_des + J*q_current
    YMatrix q_mat = vectorToColumnMatrix(q_current);
    YMatrix Jq = J * q_mat;

    std::vector<double> beq(6, 0.0);
    for (int i = 0; i < 6; ++i) {
        beq[i] = Delta_x_des[i] + Jq[i][0];
    }

    QPSolver::QPResult result = qp_solver.solve(H, f, J, beq, lb, ub, QP_tolerance);

    if (!result.success) {
        std::cerr << "[QP FAILURE]\n";
        std::cerr << "  exitflag     : " << result.exitflag << " (debug)\n";
        std::cerr << "  status_val   : " << result.status << "\n";
        if (!result.status_string.empty())
            std::cerr << "  status_str   : " << result.status_string << "\n";
        std::cerr << "  iter         : " << result.iter << "\n";
        std::cerr << "  obj_val      : " << result.objective_value << "\n";

        double min_width = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < numOfAxis; ++i) min_width = std::min(min_width, ub[i] - lb[i]);
        std::cerr << "  min(ub-lb)   : " << min_width << "\n";

        double dx_norm2 = 0.0;
        for (double v : Delta_x_des) dx_norm2 += v*v;
        std::cerr << "  ||Delta_x||  : " << std::sqrt(dx_norm2) << "\n";
        std::cerr << "  has_prev     : " << (has_prev ? 1 : 0) << "\n";
        std::cerr << "---------------------------------\n";

        return q_current;
    }

#elif Y2_IK_SOLVER_MODE == Y2_IK_SOLVER_DLS
    const double damping = std::max(std::abs(lambda), 1e-9);
    YMatrix task_metric = (J * J.transpose()) + (YMatrix::identity(6) * (damping * damping));
    YMatrix delta_q_mat = J.transpose() * task_metric.inverse() * vectorToColumnMatrix(Delta_x_des);

    std::vector<double> q_next = q_current;
    for (size_t i = 0; i < numOfAxis; ++i) {
        q_next[i] += delta_q_mat[i][0];
    }

    q_next = clampToBounds(q_next, lb, ub);
#else
#error "Unsupported Y2_IK_SOLVER_MODE"
#endif

    // 7) Update history
    q_prev = q_current;
    has_prev = true;

#if Y2_IK_SOLVER_MODE == Y2_IK_SOLVER_QP
    return result.solution;
#else
    return q_next;
#endif
}

// Utility
double Kinematics::roundToNthDecimal(double value, int n) {
    const double m = std::pow(10.0, static_cast<double>(n));
    return std::round(value * m) / m;
}

void Kinematics::setControlGains(double kp_pos, double kp_rot) {
    Kp_pos = kp_pos;
    Kp_rot = kp_rot;
}

void Kinematics::setQPWeights(double omega_p_val, double alpha_val, double lambda_val) {
    omega_p = omega_p_val;
    alpha   = alpha_val;
    lambda  = lambda_val;
}

void Kinematics::setJointLimits(const std::vector<double>& q_min_in,
                                const std::vector<double>& q_max_in,
                                const std::vector<double>& qd_min_in,
                                const std::vector<double>& qd_max_in) {
    if (q_min_in.size() != numOfAxis || q_max_in.size() != numOfAxis ||
        qd_min_in.size() != numOfAxis || qd_max_in.size() != numOfAxis) {
        throw std::invalid_argument("Joint limits must have " + std::to_string(numOfAxis) + " elements each");
    }
    q_min  = q_min_in;
    q_max  = q_max_in;
    qd_min = qd_min_in;
    qd_max = qd_max_in;
}

void Kinematics::setAccelLimits(const std::vector<double>& a_min_in,
                                const std::vector<double>& a_max_in) {
    if (a_min_in.size() != numOfAxis || a_max_in.size() != numOfAxis) {
        throw std::invalid_argument("Acceleration limits must have " + std::to_string(numOfAxis) + " elements each");
    }
    a_min = a_min_in;
    a_max = a_max_in;
}

void Kinematics::setPrevQ(const std::vector<double>& q_prev_in) {
    if (q_prev_in.size() != numOfAxis) {
        throw std::invalid_argument("q_prev must have " + std::to_string(numOfAxis) + " elements");
    }
    q_prev = q_prev_in;
    has_prev = true;
}

void Kinematics::printPose(const YMatrix& pose, const std::string& label) {
    std::cout << label << " Pose:\n";
    for (const auto& row : pose) {
        for (const auto& val : row) std::cout << val << " ";
        std::cout << "\n";
    }
}
