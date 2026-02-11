////////////////////////////////////////////////////////////
// Kinematics.h
// Kinematics functions
////////////////////////////////////////////////////////////
////////////////// Ver 2.00 ////////////////////////////////
////////////////////////////////////////////////////////////

#ifndef RTDE_HANDARM2_KINEMATIC_FUNC_H_
#define RTDE_HANDARM2_KINEMATIC_FUNC_H_

#include "Arm_class.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
// #include "NRS_yaml_location.h"

using namespace Eigen;

#ifndef ZERO_THRESH
#define ZERO_THRESH 1e-8
#endif

#ifndef SIGN
#define SIGN(x) (((x) > 0) - ((x) < 0))
#endif

// 전역 변수는 extern 선언만
extern std::ifstream Kfin1;
extern YAML::Node KNRS_Fcon_setting;

struct Quaternion { double w, x, y, z; };

typedef class Kinematic_func
{
private:
    // trigonometric vars
    double s1, c1, s2, c2, s3, c3, s4, c4, s5, c5, s6, c6;
    double s23, c23, s34, c34, s234, c234;
    // TCP transform offset
    Matrix4d Ycontact_EE2TCP;
    double Ycontact_TCP_pos[3];
    // for rotation continuity
    bool R2E_init_flag;
    Vector3d R2E_pre_rpy;

public:
    Kinematic_func();

    // ROS1 시그니처 유지
    void iForwardK_P(VectorXd &q, Vector3d &x, double endlength = 0);
    void iForwardK_T(VectorXd &q, Matrix4d &T, double endlength = 0);

    void ForwardK_P(CArm *A);
    void ForwardK_T(CArm *A);
    void ForwardK_Td(CArm *A);
    void Ycontact_ForwardK_T(CArm *A);
    void Rotation2EulerAngle(CArm *A);
    void Quaternion2Rotation(CArm *A);
    void iRotation2EulerAngle(Matrix3d &R, Vector3d &th);
    void Rotation2RPY(CArm *A);
    void EulerAngle2Rotation(Matrix3d &R, Vector3d &th);
    Vector3d VR_Rot2RPY(const Matrix3d &rotationMatrix);

    int InverseK(CArm *qA);
    int Ycontact_InverseK(CArm *qA);
    int InverseK_min(CArm *A);
    int Ycontact_InverseK_min(CArm *A);

    void Jacobian(CArm *A);
    void Jacobian_p(CArm *A);
    void Jacobian_w(CArm *A);

    int sgn(double x);

    Matrix3d RotX(double th);
    Matrix3d RotY(double th);
    Matrix3d RotZ(double th);

    Matrix3d angle_axis_representation(Vector3d rot_axis, double rot_angle);
    Matrix3d Qua2Rot(double w, double x, double y, double z);
    Quaterniond Rot2Qua(const Matrix3d &rotationMatrix);
} AKfun;

#endif  // RTDE_HANDARM2_KINEMATIC_FUNC_H_