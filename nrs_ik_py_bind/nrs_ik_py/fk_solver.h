#pragma once
#include <utility>
#include <Eigen/Dense>
#include "Arm_class.h"
#include "Kinematics.h"

// IK 쪽과 동일한 구조체 (line_no는 옵션)
struct PoseRPY {
    int    line_no{0};
    double x{0.0}, y{0.0}, z{0.0};
    double r{0.0}, p{0.0}, yaw{0.0};
};

class FKSolver {
public:
    // tool_z: EE->TCP z 오프셋 [m], use_degrees: 기본 입·출력 각 단위를 deg로 쓸지 여부(옵션)
    FKSolver(double tool_z, bool use_degrees = false);

    // FK 계산: q(6x1) -> PoseRPY
    // as_degrees=True면 roll/pitch/yaw를 deg로 반환
    std::pair<bool, PoseRPY> compute(const Eigen::Matrix<double,6,1>& q, bool as_degrees = false);

    // (옵션) 동일 동작의 alias
    PoseRPY transform(const Eigen::Matrix<double,6,1>& q, bool as_degrees = false);

private:
    double         tool_z_;
    bool           use_degrees_;
    CArm           arm_;
    Kinematic_func kin_;
};
