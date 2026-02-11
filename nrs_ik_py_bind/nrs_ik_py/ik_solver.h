#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include "Arm_class.h"
#include "Kinematics.h"

// PoseRPY 구조체
struct PoseRPY {
    size_t line_no;
    double x, y, z;
    double r, p, yaw;
};

class IKSolver {
public:
    IKSolver(double tool_z, bool use_degrees);

    bool load_txt(const std::string& path);
    std::pair<bool, Eigen::Matrix<double,6,1>> compute(const PoseRPY& P);

    // 새로 추가: 인덱스로 바로 접근
    std::pair<bool, Eigen::Matrix<double,6,1>> compute_idx(int idx);

    const std::vector<PoseRPY>& getPoses() const { return poses_; }

private:
    double tool_z_;
    bool use_degrees_;
    Kinematic_func kin_;
    CArm arm_;
    std::vector<PoseRPY> poses_;
};
