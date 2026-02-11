#include "ik_solver.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;

IKSolver::IKSolver(double tool_z, bool use_degrees)
: tool_z_(tool_z), use_degrees_(use_degrees) {
    arm_.qc = Eigen::Vector<double, 6>::Zero();
    arm_.qd = Eigen::Vector<double, 6>::Zero();
    arm_.q  = Eigen::Vector<double, 6*8>::Zero();
    arm_.Td = Matrix4d::Identity();
    arm_.Tc = Matrix4d::Identity();
    arm_.R2E_init_flag = false;
    arm_.pre_thc = Eigen::Vector3d::Zero();
    arm_.thc     = Eigen::Vector3d::Zero();
}

bool IKSolver::load_txt(const std::string& path) {
    std::ifstream infile(path);
    if (!infile.is_open()) return false;

    std::string line;
    size_t line_no = 0;
    poses_.clear();

    while (std::getline(infile, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::vector<double> cols{ std::istream_iterator<double>(iss), std::istream_iterator<double>() };
        if (cols.size() < 6) continue;

        PoseRPY p;
        p.line_no = line_no;
        p.x = cols[0]; p.y = cols[1]; p.z = cols[2];
        p.r = cols[3]; p.p = cols[4]; p.yaw = cols[5];

        if (use_degrees_) {
            constexpr double D2R = M_PI/180.0;
            p.r   *= D2R;
            p.p   *= D2R;
            p.yaw *= D2R;
        }
        poses_.push_back(p);
    }
    return !poses_.empty();
}

std::pair<bool, Eigen::Matrix<double,6,1>> IKSolver::compute(const PoseRPY& P) {
    Matrix3d R;
    Vector3d th; th << P.r, P.p, P.yaw;
    kin_.EulerAngle2Rotation(R, th);

    Matrix4d Td_tcp = Matrix4d::Identity();
    Td_tcp.block<3,3>(0,0) = R;
    Td_tcp(0,3) = P.x; Td_tcp(1,3) = P.y; Td_tcp(2,3) = P.z;

    Matrix4d EE2TCP = Matrix4d::Identity();
    EE2TCP(2,3) = tool_z_;

    arm_.Td = Td_tcp * EE2TCP.inverse();
    arm_.qc = arm_.qd;

    int nsol = kin_.InverseK_min(&arm_);
    if (nsol <= 0) return {false, Eigen::Matrix<double,6,1>::Zero()};

    Eigen::Matrix<double,6,1> q;
    for (int i = 0; i < 6; ++i) q(i) = arm_.qd(i);
    return {true, q};
}

// 새로 추가: idx 버전
std::pair<bool, Eigen::Matrix<double,6,1>> IKSolver::compute_idx(int idx) {
    if (idx < 0 || idx >= (int)poses_.size())
        return {false, Eigen::Matrix<double,6,1>::Zero()};
    return compute(poses_[idx]);
}
