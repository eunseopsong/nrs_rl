/*
    This code is publicly available on GitHub.
    You are free to use, modify, and distribute it.
    However, you must clearly credit the source as follows:

    Source: -
    Author: jaeyun Sim
    Email: wodbs02221@gmail.com
*/

#pragma once

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <limits>
#include <stdexcept>   // 추가: invalid_argument, runtime_error
#include <algorithm>   // 추가: std::min

struct RPY {
    double roll, pitch, yaw;
    RPY(double r = 0.0, double p = 0.0, double y = 0.0) : roll(r), pitch(p), yaw(y) {}
    RPY(const std::vector<double>& rpy) : roll(rpy[0]), pitch(rpy[1]), yaw(rpy[2]) {}
};

struct Quaternion {
    double w, x, y, z;
    Quaternion(double w = 1.0, double x = 0.0, double y = 0.0, double z = 0.0) : w(w), x(x), y(y), z(z) {}
    Quaternion(const std::vector<double>& q) : w(q[0]), x(q[1]), y(q[2]), z(q[3]) {}
};

// 새로운 SpatialAngle 구조체
struct SpatialAngle {
    double x, y, z;  // 각 축의 회전 성분 (축 방향 × 각도)
    
    // 기본 생성자
    SpatialAngle(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
    
    // 벡터로부터 생성
    SpatialAngle(const std::vector<double>& vec) {
        if (vec.size() >= 3) {
            x = vec[0]; y = vec[1]; z = vec[2];
        } else {
            x = y = z = 0.0;
        }
    }
    
    // 벡터로 변환
    std::vector<double> toVector() const {
        return {x, y, z};
    }
    
    // 회전 각도 크기 반환
    double magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    // 회전축 반환 (단위벡터)
    std::vector<double> axis() const {
        double mag = magnitude();
        if (mag < 1e-10) {
            return {0.0, 0.0, 1.0};  // 기본축 Z
        }
        return {x/mag, y/mag, z/mag};
    }
    
    // 회전 각도 반환 (라디안)
    double angle() const {
        return magnitude();
    }
    
    // 정규화 (단위 회전축으로 만들기)
    SpatialAngle normalized() const {
        double mag = magnitude();
        if (mag < 1e-10) {
            return SpatialAngle(0.0, 0.0, 0.0);
        }
        return SpatialAngle(x/mag, y/mag, z/mag);
    }
    
    // 연산자 오버로딩
    SpatialAngle operator+(const SpatialAngle& other) const {
        return SpatialAngle(x + other.x, y + other.y, z + other.z);
    }
    
    SpatialAngle operator-(const SpatialAngle& other) const {
        return SpatialAngle(x - other.x, y - other.y, z - other.z);
    }
    
    SpatialAngle operator*(double scalar) const {
        return SpatialAngle(x * scalar, y * scalar, z * scalar);
    }
    
    SpatialAngle operator/(double scalar) const {
        if (std::abs(scalar) < 1e-10) {
            throw std::invalid_argument("Division by zero");
        }
        return SpatialAngle(x / scalar, y / scalar, z / scalar);
    }
    
    // 내적
    double dot(const SpatialAngle& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // 외적
    SpatialAngle cross(const SpatialAngle& other) const {
        return SpatialAngle(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    // 두 spatial angle 사이의 각도
    double angleBetween(const SpatialAngle& other) const {
        double dot_product = dot(other);
        double mag_product = magnitude() * other.magnitude();
        if (mag_product < 1e-10) return 0.0;
        
        double cos_angle = std::max(-1.0, std::min(1.0, dot_product / mag_product));
        return std::acos(cos_angle);
    }
    
    // 출력
    void print() const {
        std::cout << "SpatialAngle: [" << x << ", " << y << ", " << z << "]" 
                  << " (angle: " << angle() * 180.0 / M_PI << "°)" << std::endl;
    }
    
    // 0인지 확인
    bool isZero(double tolerance = 1e-10) const {
        return magnitude() < tolerance;
    }
};

class YMatrix  {
private:

    /*** Variables ***/
    std::vector<std::vector<double>> data;

    /* 쿼터니언 적분용 static 변수들 (상태 저장용) */ 
    static Quaternion last_quaternion;
    static std::vector<double> last_rpy;
    static std::vector<double> filtered_angular_velocity;
    static bool integrator_initialized;
    static double integration_dt;
    static double filter_alpha;

    /*** Functions ***/
    
    /* 각도를 [-π, π] 범위로 정규화 */
    static double normalizeAngle(double angle);
    
    /* 벡터를 정규화 */
    static std::vector<double> normalize(const std::vector<double>& vec);

    /* 쿼터니언 연속성 보장 */
    static Quaternion ensureQuaternionContinuity(const Quaternion& current, const Quaternion& previous);
    
    /* 쿼터니언 차이로부터 각속도 계산 */
    static std::vector<double> computeAngularVelocityFromQuaternions(
        const Quaternion& q_prev, const Quaternion& q_curr, double dt);
    
    /* Euler Rate Equations 적용 */
    static std::vector<double> applyEulerRateEquations(
        const std::vector<double>& rpy, const std::vector<double>& omega, double dt);
    
    /* 로우패스 필터 적용 */
    static std::vector<double> applyLowPassFilter(
        const std::vector<double>& new_omega, const std::vector<double>& old_omega, double alpha);
    
public:

    std::vector<std::vector<double>>::iterator begin() { return data.begin(); }
    std::vector<std::vector<double>>::iterator end() { return data.end(); }
    std::vector<std::vector<double>>::const_iterator begin() const { return data.begin(); }
    std::vector<std::vector<double>>::const_iterator end() const { return data.end(); }

    /* 행, 열 크기를 설정한 경우 */
    YMatrix(size_t rows, size_t cols) : data(rows, std::vector<double>(cols, 0.0)) {}
    
    /* 행, 열 크기를 설정하지 않은 경우 */
    YMatrix(std::initializer_list<std::initializer_list<double>> list);
    
    /* Index access (Matrix 객체를 배열처럼 matrix[i]로 접근할 수 있게 만들기, 연산자 오버로딩 사용용) */ 
    std::vector<double>& operator[](size_t i) { return data[i]; }
    const std::vector<double>& operator[](size_t i) const { return data[i]; }
    
    /****** Matrix Basic ******/

    /* Size return function */ 
    size_t rows() const { return data.size(); }
    size_t cols() const { return data[0].size(); }
    
    /* 행렬 덧셈 */ 
    YMatrix operator+(const YMatrix& other) const;
    
    /* 행렬 뺄셈 🔹 추가 */
    YMatrix operator-(const YMatrix& other) const;
    
    /* 행렬 곱셈 */ 
    YMatrix operator*(const YMatrix& other) const;

    /* 스칼라 곱셈 연산자 (행렬 * 스칼라) */
    YMatrix operator*(double scalar) const;
    
    /* 스칼라 나눗셈 연산자 */
    YMatrix operator/(double scalar) const;

    /* 전치행렬 */ 
    YMatrix transpose() const;

    /* 역행렬 계산 (가우스-조던 소거법 사용) */
    YMatrix inverse() const;

    /* Extract function */
    YMatrix extract(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols) const;

    /* Insert function */
    void insert(size_t start_row, size_t start_col, const YMatrix& others);

    /* Matrix to std::vector */
    std::vector<double> toVector() const;

    /* Vertical Append function */ 
    void appendV(const YMatrix& object);

    /* Horizontal Append function */ 
    void appendH(const YMatrix& object);

    /* 행렬 크기 조정 (기존 데이터 보존) */
    void resize(size_t new_rows, size_t new_cols);
    
    /* 출력 */
    void print() const;

    /* 행렬을 txt 파일로 저장 */
    void saveToFile(const std::string& filePath, int precision = 10) const;
    
    /* txt 파일로부터 행렬 로드 */
    static YMatrix loadFromFile(const std::string& filePath);

    /****** Rotation Transformation ******/

    /* RPY(Roll-Pitch-Yaw)로부터 회전 행렬 생성 */
    static YMatrix fromRPY(double roll, double pitch, double yaw);
    
    /* Quaternion(w,x,y,z)으로부터 회전 행렬 생성 */
    static YMatrix fromQuaternion(double w, double x, double y, double z);
    
    /* Angle-Axis(axis_x, axis_y, axis_z, angle)로부터 회전 행렬 생성 */
    static YMatrix fromAngleAxis(double axis_x, double axis_y, double axis_z, double angle);
    
    /* 회전 행렬을 RPY로 변환 (roll, pitch, yaw 순서로 반환) */
    std::vector<double> toRPY() const;
    
    /* 회전 행렬을 Quaternion으로 변환 (w,x,y,z 순서로 반환) */
    std::vector<double> toQuaternion() const;
    
    /* 회전 행렬을 Angle-Axis로 변환 (axis_x, axis_y, axis_z, angle 순서로 반환) */
    std::vector<double> toAngleAxis() const;
    
    /* RPY to Quaternion */
    static Quaternion rpyToQuaternion(double roll, double pitch, double yaw);

    /* Quaternion to RPY */
    static RPY quaternionToRPY(const Quaternion& q);

    /* 두 쿼터니안 사이의 각도 계산 */
    static double angleBetweenQuaternions(const Quaternion& q1, const Quaternion& q2);

    /* 단위 행렬 생성 */
    static YMatrix identity(size_t size);
    
    /* 회전 행렬인지 확인 (정규직교 행렬이고 det=1인지 확인) */
    bool isRotationMatrix() const;
    
    /* 행렬식 계산 (3x3만 지원) */
    double determinant() const;
    
    /* 크기(norm) 계산 */
    double norm() const;

    /*** Advanced Quaternion to RPY (Using Quaternion Integral) ***/

    /* 쿼터니언 적분 방식으로 RPY 변환 (연속성 보장) */
    static RPY quaternionToRPYSmooth(const Quaternion& q, double sampling_time = 0.002);
    
    /* 쿼터니언 시퀀스를 연속적인 RPY 시퀀스로 변환 */
    static std::vector<std::vector<double>> quaternionSequenceToRPYSmooth(
        const std::vector<Quaternion>& quat_sequence, double dt = 0.002);
    
    /* 적분기 설정 함수들 */
    static void setIntegrationParams(double dt = 0.002, double filter_alpha = 0.8);
    static void resetIntegrator();
    
    /* 수동 적분 (이전 상태를 직접 제공) */
    static std::vector<double> quaternionToRPYIntegrated(
        const Quaternion& current_q, 
        const Quaternion& previous_q,
        const std::vector<double>& previous_rpy,
        double dt = 0.002);

    /****** SpatialAngle Transformation Functions ******/
    
    /* 회전 행렬을 SpatialAngle로 변환 */
    SpatialAngle toSpatialAngle() const;

    /* SpatialAngle로부터 회전 행렬 생성 */
    static YMatrix fromSpatialAngle(const SpatialAngle& spatial_angle);
    
    /* SpatialAngle 변환 간 연속성 보장 함수들 */ 
    SpatialAngle toSpatialAngleContinuous(const SpatialAngle& reference) const;
    static YMatrix fromSpatialAngleContinuous(const SpatialAngle& spatial_angle, const YMatrix& reference_R);
    static SpatialAngle ensureSpatialAngleContinuity(const SpatialAngle& reference, const SpatialAngle& target);
    static YMatrix normalizeRotationMatrix(const YMatrix& R);


    /* SpatialAngle을 Quaternion으로 변환 */
    static Quaternion spatialAngleToQuaternion(const SpatialAngle& spatial_angle);

    /* Quaternion을 SpatialAngle로 변환 */
    static SpatialAngle quaternionToSpatialAngle(const Quaternion& q);

    /* 두 회전 행렬 사이의 SpatialAngle 오차 계산 */
    static SpatialAngle spatialAngularError(const YMatrix& target_R, const YMatrix& current_R);

    /* SpatialAngle 보간 (SLERP 기반) */
    static SpatialAngle interpolateSpatialAngle(
        const SpatialAngle& spatial_angle1,
        const SpatialAngle& spatial_angle2,
        double t);

    /* RPY에서 SpatialAngle로 변환 */
    static SpatialAngle rpyToSpatialAngle(const RPY& rpy);

    /* SpatialAngle에서 RPY로 변환 */
    static RPY spatialAngleToRPY(const SpatialAngle& spatial_angle);
    
};
