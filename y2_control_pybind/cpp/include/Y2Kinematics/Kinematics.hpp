#pragma once

#include "Y2Matrix/YMatrix.hpp"
#include "Y2Kinematics/QP_solver.hpp"

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

#define Y2_IK_SOLVER_QP  0
#define Y2_IK_SOLVER_DLS 1

#ifndef Y2_IK_SOLVER_MODE
#define Y2_IK_SOLVER_MODE Y2_IK_SOLVER_DLS
#endif

class Kinematics {
public:
    Kinematics(double SamplingTime_ = 0.01,
               size_t numOfAxis_ = 7,
               const YMatrix& EE2TCP_ = YMatrix::identity(4));

    // Forward Kinematics (Based on End-Effector)
    virtual YMatrix forwardKinematics(const std::vector<double>& q) = 0;

    // Jacobian Calculation (Based on End-Effector)
    virtual YMatrix calculateJacobian(const std::vector<double>& q) = 0;

    // IK solver (Based on End-Effector)
    // Solver backend is selected at compile time with Y2_IK_SOLVER_MODE.
    std::vector<double> solve_IK(const std::vector<double>& q_current,
                                 const YMatrix& target_HTM);

    // Utility
    static double roundToNthDecimal(double value, int n);

    void setControlGains(double kp_pos, double kp_rot);
    void setQPWeights(double omega_p_val, double alpha_val, double lambda_val);

    void setJointLimits(const std::vector<double>& q_min,
                        const std::vector<double>& q_max,
                        const std::vector<double>& qd_min,
                        const std::vector<double>& qd_max);

    void setAccelLimits(const std::vector<double>& a_min,
                        const std::vector<double>& a_max);

    void setPrevQ(const std::vector<double>& q_prev_in);

    void printPose(const YMatrix& pose, const std::string& label);

protected:
    double dt = 0.01;
    size_t numOfAxis = 7;
    YMatrix EE2TCP = YMatrix::identity(4);

    int QP_precision = 4;
    double QP_tolerance = 1e-4; // will be set in ctor from QP_precision

    // task gains
    double Kp_pos = 100.0;
    double Kp_rot = 10.0;

    // weights
    double omega_p = 1.0;
    double alpha   = 0.1;
    double lambda  = 0.001;

    // limits
    std::vector<double> q_min, q_max;
    std::vector<double> qd_min, qd_max;
    std::vector<double> a_min, a_max;

    // history
    std::vector<double> q_prev;
    bool has_prev = false;

private:
    QPSolver qp_solver;
};
