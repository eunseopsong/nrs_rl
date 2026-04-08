#include "Y2Kinematics/KinematicsUR10e.hpp"

KinematicsUR10e::KinematicsUR10e(double SamplingTime_, size_t numOfAxis_, const YMatrix& EE2TCP_) 
: Kinematics(SamplingTime_, numOfAxis_, EE2TCP_) 
{
    // Set QP - parameters
    setControlGains(1.0, 1.0);
    setQPWeights(1.0, 0.1, 0.03);

    // Set joint limits
    setJointLimits(q_min, q_max, qd_min, qd_max);
    setAccelLimits(qdd_min,qdd_max);
}


// Forward Kinematics implementation based on MATLAB UR10_FK06
YMatrix KinematicsUR10e::forwardKinematics(const std::vector<double>& q) {
    if (q.size() != numOfAxis) {
        throw std::invalid_argument("Joint angles must have 6 elements");
    }
    
    double th1 = q[0], th2 = q[1], th3 = q[2];
    double th4 = q[3], th5 = q[4], th6 = q[5];
    
    // Pre-calculate trigonometric functions
    double s1 = sin(th1), c1 = cos(th1);
    double s2 = sin(th2), c2 = cos(th2);
    double s3 = sin(th3), c3 = cos(th3);
    double s4 = sin(th4), c4 = cos(th4);
    double s5 = sin(th5), c5 = cos(th5);
    double s6 = sin(th6), c6 = cos(th6);
    
    // Combined angles
    double th234 = th2 + th3 + th4;
    double s234 = sin(th234), c234 = cos(th234);
    
    double th23 = th2 + th3;
    double s23 = sin(th23), c23 = cos(th23);
    
    YMatrix T(4, 4);
    
    // Row 1 - Based on MATLAB T06(1,:)
    T[0][0] = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6;
    T[0][1] = -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6;
    T[0][2] = c5 * s1 - c234 * c1 * s5;
    T[0][3] = d6 * (c5 * s1 - c234 * c1 * s5) + d4 * s1 + a3 * c23 * c1 + a2 * c1 * c2 + d5 * s234 * c1;
    
    // Row 2 - Based on MATLAB T06(2,:) 
    T[1][0] = -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6;
    T[1][1] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1;
    T[1][2] = -c1 * c5 - c234 * s1 * s5;
    T[1][3] = a3 * c23 * s1 - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + a2 * c2 * s1 + d5 * s234 * s1;
    
    // Row 3 - Based on MATLAB T06(3,:)
    T[2][0] = c234 * s6 + s234 * c5 * c6;
    T[2][1] = c234 * c6 - s234 * c5 * s6;
    T[2][2] = -s234 * s5;
    T[2][3] = d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5;
    
    // Row 4 - Homogeneous coordinates
    T[3][0] = 0; T[3][1] = 0; T[3][2] = 0; T[3][3] = 1;
    
    return T * EE2TCP;
}

// Jacobian calculation based on MATLAB UR10_jacobian
YMatrix KinematicsUR10e::calculateJacobian(const std::vector<double>& q) {
    if (q.size() != numOfAxis) {
        throw std::invalid_argument("Joint angles must have 6 elements");
    }
    
    double th1 = q[0], th2 = q[1], th3 = q[2];
    double th4 = q[3], th5 = q[4], th6 = q[5];
    
    // Pre-calculate trigonometric functions
    double s1 = sin(th1), c1 = cos(th1);
    double s2 = sin(th2), c2 = cos(th2);
    double s3 = sin(th3), c3 = cos(th3);
    double s4 = sin(th4), c4 = cos(th4);
    double s5 = sin(th5), c5 = cos(th5);
    double s6 = sin(th6), c6 = cos(th6);
    
    // Combined angles
    double th234 = th2 + th3 + th4;
    double s234 = sin(th234), c234 = cos(th234);
    
    double th23 = th2 + th3;
    double s23 = sin(th23), c23 = cos(th23);
    
    // Combined sine terms for d6 calculations
    double sin_234_minus_5 = sin(th234 - th5);
    double sin_234_plus_5 = sin(th234 + th5);
    
    YMatrix J(6, numOfAxis);
    
    // First row (Jv_x - x-direction linear velocity)
    J[0][0] = d6 * (c1 * c5 + c234 * s1 * s5) + d4 * c1 - a3 * c23 * s1 - a2 * c2 * s1 - d5 * s234 * s1;
    
    J[0][1] = -c1 * (a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5);
    
    J[0][2] = c1 * (d5 * c234 - a3 * s23 + d6 * s234 * s5);
    
    J[0][3] = c1 * (d5 * c234 + d6 * s234 * s5);
    
    J[0][4] = -d6 * s1 * s5 - d6 * c234 * c1 * c5;
    
    J[0][5] = 0;
    
    // Second row (Jv_y - y-direction linear velocity)
    J[1][0] = d6 * (c5 * s1 - c234 * c1 * s5) + d4 * s1 + a3 * c23 * c1 + a2 * c1 * c2 + d5 * s234 * c1;
    
    J[1][1] = -s1 * (a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5);
    
    J[1][2] = s1 * (d5 * c234 - a3 * s23 + d6 * s234 * s5);
    
    J[1][3] = s1 * (d5 * c234 + d6 * s234 * s5);
    
    J[1][4] = d6 * c1 * s5 - d6 * c234 * c5 * s1;
    
    J[1][5] = 0;
    
    // Third row (Jv_z - z-direction linear velocity)
    J[2][0] = 0;
    
    J[2][1] = a3 * c23 - (d6 * sin_234_plus_5) / 2.0 + a2 * c2 + (d6 * sin_234_minus_5) / 2.0 + d5 * s234;
    
    J[2][2] = a3 * c23 - (d6 * sin_234_plus_5) / 2.0 + (d6 * sin_234_minus_5) / 2.0 + d5 * s234;
    
    J[2][3] = d5 * s234 - d6 * c234 * s5;
    
    J[2][4] = -d6 * s234 * c5;
    
    J[2][5] = 0;
    
    // Fourth row (Jw_x - x-direction angular velocity)
    J[3][0] = 0;
    J[3][1] = s1;
    J[3][2] = s1;
    J[3][3] = s1;
    J[3][4] = s234 * c1;
    J[3][5] = c5 * s1 - c234 * c1 * s5;
    
    // Fifth row (Jw_y - y-direction angular velocity)
    J[4][0] = 0;
    J[4][1] = -c1;
    J[4][2] = -c1;
    J[4][3] = -c1;
    J[4][4] = s234 * s1;
    J[4][5] = -c1 * c5 - c234 * s1 * s5;
    
    // Sixth row (Jw_z - z-direction angular velocity)
    J[5][0] = 1;
    J[5][1] = 0;
    J[5][2] = 0;
    J[5][3] = 0;
    J[5][4] = -c234;
    J[5][5] = -s234 * s5;
    
    return J;
}