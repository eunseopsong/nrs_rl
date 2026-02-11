#include "fk_solver.h"
#include <cmath>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;

FKSolver::FKSolver(double tool_z, bool use_degrees)
: tool_z_(tool_z), use_degrees_(use_degrees)
{
    // Arm_class 내부는 IK에서 이미 초기화되지만, FK에서도 안전하게 기본값 세팅
    arm_.qc = Eigen::Vector<double, 6>::Zero();
    arm_.Tc = Matrix4d::Identity();
}

std::pair<bool, PoseRPY>
FKSolver::compute(const Eigen::Matrix<double,6,1>& q, bool as_degrees)
{
    // 1) EE 포즈 계산 (Kinematics의 독립 함수 사용)
    VectorXd qv(6);
    for (int i=0; i<6; ++i) qv(i) = q(i);

    Matrix4d T_ee = Matrix4d::Identity();
    kin_.iForwardK_T(qv, T_ee, /*endlength=*/0.0);   // EE 기준 FK

    // 2) TCP 오프셋(EE->TCP) 적용: IK와 동일 기준 유지
    Matrix4d EE2TCP = Matrix4d::Identity();
    EE2TCP(2,3) = tool_z_;
    Matrix4d T_tcp = T_ee * EE2TCP;

    // 3) ZYX(Rz*Ry*Rx) 정의로 RPY 추출 → IK와 일치
    Matrix3d R = T_tcp.block<3,3>(0,0);
    Vector3d rpy;
    kin_.iRotation2EulerAngle(R, rpy);  // roll=rpy(0), pitch=rpy(1), yaw=rpy(2)

    PoseRPY out;
    out.x   = T_tcp(0,3);
    out.y   = T_tcp(1,3);
    out.z   = T_tcp(2,3);
    out.r   = rpy(0);
    out.p   = rpy(1);
    out.yaw = rpy(2);

    if (as_degrees || use_degrees_) {
        constexpr double R2D = 180.0 / M_PI;
        out.r   *= R2D;
        out.p   *= R2D;
        out.yaw *= R2D;
    }
    return {true, out};
}

PoseRPY FKSolver::transform(const Eigen::Matrix<double,6,1>& q, bool as_degrees)
{
    auto res = compute(q, as_degrees);
    return res.first ? res.second : PoseRPY{};
}
