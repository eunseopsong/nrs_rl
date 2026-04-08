#include "Y2Matrix/YMatrix.hpp"

/* RPY(Roll-Pitch-Yaw)로부터 회전 행렬 생성 */
YMatrix YMatrix::fromRPY(double roll, double pitch, double yaw) {
    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);
    
    YMatrix R(3, 3);
    
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R[0][0] = cy * cp;
    R[0][1] = cy * sp * sr - sy * cr;
    R[0][2] = cy * sp * cr + sy * sr;
    
    R[1][0] = sy * cp;
    R[1][1] = sy * sp * sr + cy * cr;
    R[1][2] = sy * sp * cr - cy * sr;
    
    R[2][0] = -sp;
    R[2][1] = cp * sr;
    R[2][2] = cp * cr;
    
    return R;
}

/* Quaternion(w,x,y,z)으로부터 회전 행렬 생성 */
YMatrix YMatrix::fromQuaternion(double w, double x, double y, double z) {
    // 쿼터니언 정규화
    double norm = sqrt(w*w + x*x + y*y + z*z);
    w /= norm; x /= norm; y /= norm; z /= norm;
    
    YMatrix R(3, 3);
    
    // 쿼터니언을 회전 행렬로 변환
    R[0][0] = 1 - 2*(y*y + z*z);
    R[0][1] = 2*(x*y - w*z);
    R[0][2] = 2*(x*z + w*y);
    
    R[1][0] = 2*(x*y + w*z);
    R[1][1] = 1 - 2*(x*x + z*z);
    R[1][2] = 2*(y*z - w*x);
    
    R[2][0] = 2*(x*z - w*y);
    R[2][1] = 2*(y*z + w*x);
    R[2][2] = 1 - 2*(x*x + y*y);
    
    return R;
}

/* Angle-Axis(axis_x, axis_y, axis_z, angle)로부터 회전 행렬 생성 */
YMatrix YMatrix::fromAngleAxis(double axis_x, double axis_y, double axis_z, double angle) {
    // 축 벡터 정규화
    double norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    if (norm < 1e-10) {
        return identity(3);  // 영벡터인 경우 단위행렬 반환
    }
    
    axis_x /= norm;
    axis_y /= norm;
    axis_z /= norm;
    
    double c = cos(angle);
    double s = sin(angle);
    double one_minus_c = 1 - c;
    
    YMatrix R(3, 3);
    
    // Rodrigues' rotation formula
    R[0][0] = c + axis_x*axis_x*one_minus_c;
    R[0][1] = axis_x*axis_y*one_minus_c - axis_z*s;
    R[0][2] = axis_x*axis_z*one_minus_c + axis_y*s;
    
    R[1][0] = axis_y*axis_x*one_minus_c + axis_z*s;
    R[1][1] = c + axis_y*axis_y*one_minus_c;
    R[1][2] = axis_y*axis_z*one_minus_c - axis_x*s;
    
    R[2][0] = axis_z*axis_x*one_minus_c - axis_y*s;
    R[2][1] = axis_z*axis_y*one_minus_c + axis_x*s;
    R[2][2] = c + axis_z*axis_z*one_minus_c;
    
    return R;
}

/* 회전 행렬을 RPY로 변환 */
std::vector<double> YMatrix::toRPY() const {
    if (rows() != 3 || cols() != 3) {
        throw std::invalid_argument("Matrix must be 3x3 for RPY conversion");
    }
    
    double roll, pitch, yaw;
    
    // Singularity check (gimbal lock)
    if (abs(data[2][0]) >= 0.99999) {
        // Gimbal lock case
        yaw = 0.0;  // yaw를 0으로 설정
        if (data[2][0] < 0) {
            pitch = M_PI / 2.0;
            roll = atan2(data[0][1], data[0][2]);
        } else {
            pitch = -M_PI / 2.0;
            roll = atan2(-data[0][1], data[0][2]);
        }
    } else {
        // Normal case
        pitch = asin(-data[2][0]);
        roll = atan2(data[2][1], data[2][2]);
        yaw = atan2(data[1][0], data[0][0]);
    }
    
    return {normalizeAngle(roll), normalizeAngle(pitch), normalizeAngle(yaw)};
}

/* 회전 행렬을 Quaternion으로 변환 */
std::vector<double> YMatrix::toQuaternion() const {
    if (rows() != 3 || cols() != 3) {
        throw std::invalid_argument("Matrix must be 3x3 for Quaternion conversion");
    }
    
    double w, x, y, z;
    double trace = data[0][0] + data[1][1] + data[2][2];
    
    if (trace > 0) {
        double s = sqrt(trace + 1.0) * 2; // s = 4 * w
        w = 0.25 * s;
        x = (data[2][1] - data[1][2]) / s;
        y = (data[0][2] - data[2][0]) / s;
        z = (data[1][0] - data[0][1]) / s;
    } else if ((data[0][0] > data[1][1]) && (data[0][0] > data[2][2])) {
        double s = sqrt(1.0 + data[0][0] - data[1][1] - data[2][2]) * 2; // s = 4 * x
        w = (data[2][1] - data[1][2]) / s;
        x = 0.25 * s;
        y = (data[0][1] + data[1][0]) / s;
        z = (data[0][2] + data[2][0]) / s;
    } else if (data[1][1] > data[2][2]) {
        double s = sqrt(1.0 + data[1][1] - data[0][0] - data[2][2]) * 2; // s = 4 * y
        w = (data[0][2] - data[2][0]) / s;
        x = (data[0][1] + data[1][0]) / s;
        y = 0.25 * s;
        z = (data[1][2] + data[2][1]) / s;
    } else {
        double s = sqrt(1.0 + data[2][2] - data[0][0] - data[1][1]) * 2; // s = 4 * z
        w = (data[1][0] - data[0][1]) / s;
        x = (data[0][2] + data[2][0]) / s;
        y = (data[1][2] + data[2][1]) / s;
        z = 0.25 * s;
    }
    
    return {w, x, y, z};
}

/* 회전 행렬을 Angle-Axis로 변환 */
std::vector<double> YMatrix::toAngleAxis() const {
    if (rows() != 3 || cols() != 3) {
        throw std::invalid_argument("Matrix must be 3x3 for Angle-Axis conversion");
    }
    
    double angle = acos((data[0][0] + data[1][1] + data[2][2] - 1.0) / 2.0);
    
    if (abs(angle) < 1e-6) {
        // No rotation
        return {1.0, 0.0, 0.0, 0.0};
    } else if (abs(angle - M_PI) < 1e-6) {
        // 180도 회전 - 특별한 처리 필요
        double axis_x, axis_y, axis_z;
        if (data[0][0] >= data[1][1] && data[0][0] >= data[2][2]) {
            axis_x = sqrt((data[0][0] + 1.0) / 2.0);
            axis_y = data[0][1] / (2.0 * axis_x);
            axis_z = data[0][2] / (2.0 * axis_x);
        } else if (data[1][1] >= data[2][2]) {
            axis_y = sqrt((data[1][1] + 1.0) / 2.0);
            axis_x = data[0][1] / (2.0 * axis_y);
            axis_z = data[1][2] / (2.0 * axis_y);
        } else {
            axis_z = sqrt((data[2][2] + 1.0) / 2.0);
            axis_x = data[0][2] / (2.0 * axis_z);
            axis_y = data[1][2] / (2.0 * axis_z);
        }
        return {axis_x, axis_y, axis_z, angle};
    } else {
        // 일반적인 경우
        double sin_angle = sin(angle);
        double axis_x = (data[2][1] - data[1][2]) / (2.0 * sin_angle);
        double axis_y = (data[0][2] - data[2][0]) / (2.0 * sin_angle);
        double axis_z = (data[1][0] - data[0][1]) / (2.0 * sin_angle);
        
        return {axis_x, axis_y, axis_z, angle};
    }
}

/* RPY to Quaternion */
Quaternion YMatrix::rpyToQuaternion(double roll, double pitch, double yaw) {
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    
    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    
    return q;
}

/* Quaternion to RPY */
RPY YMatrix::quaternionToRPY(const Quaternion& q) {
    RPY rpy;
    
    // Roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    rpy.roll = atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (fabs(sinp) >= 1) {
        rpy.pitch = copysign(M_PI / 2, sinp); // 특이점에서 ±90도
    } else {
        rpy.pitch = asin(sinp);
    }
    
    // Yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    rpy.yaw = atan2(siny_cosp, cosy_cosp);
    
    return rpy;
}

/* 두 쿼터니안 사이의 각도 계산 */
double YMatrix::angleBetweenQuaternions(const Quaternion& q1, const Quaternion& q2) {
    double dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
    dot = fabs(dot);  // 최단 경로 선택
    if (dot > 1.0) dot = 1.0;  // 수치 오차 방지
    
    return 2.0 * acos(dot);  // 라디안 단위
}

/* 단위 행렬 생성 */
YMatrix YMatrix::identity(size_t size) {
    YMatrix I(size, size);
    for (size_t i = 0; i < size; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

/* 회전 행렬인지 확인 */
bool YMatrix::isRotationMatrix() const {
    if (rows() != 3 || cols() != 3) {
        return false;
    }
    
    // R * R^T = I 확인
    YMatrix RT = transpose();
    YMatrix product = (*this) * RT;
    YMatrix I = identity(3);
    
    const double tolerance = 1e-6;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (abs(product[i][j] - I[i][j]) > tolerance) {
                return false;
            }
        }
    }
    
    // det(R) = 1 확인
    return abs(determinant() - 1.0) < tolerance;
}

/* 행렬식 계산 (3x3만 지원) */
double YMatrix::determinant() const {
    if (rows() != 3 || cols() != 3) {
        throw std::invalid_argument("Determinant calculation only supported for 3x3 matrices");
    }
    
    return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1]) -
           data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0]) +
           data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
}

/* 크기(norm) 계산 */
double YMatrix::norm() const {
    double sum = 0.0;
    for (const auto& row : data) {
        for (double val : row) {
            sum += val * val;
        }
    }
    return sqrt(sum);
}

/* 각도를 [-π, π] 범위로 정규화 */
double YMatrix::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

/* 벡터를 정규화 */
std::vector<double> YMatrix::normalize(const std::vector<double>& vec) {
    double norm = 0.0;
    for (double val : vec) {
        norm += val * val;
    }
    norm = sqrt(norm);
    
    std::vector<double> normalized;
    for (double val : vec) {
        normalized.push_back(val / norm);
    }
    return normalized;
}

/**** Advanced Quaternion to RPY ****/

// Static 변수 초기화
Quaternion YMatrix::last_quaternion = {1.0, 0.0, 0.0, 0.0};
std::vector<double> YMatrix::last_rpy = {0.0, 0.0, 0.0};
std::vector<double> YMatrix::filtered_angular_velocity = {0.0, 0.0, 0.0};
bool YMatrix::integrator_initialized = false;
double YMatrix::integration_dt = 0.002;
double YMatrix::filter_alpha = 0.8;

/* 쿼터니언을 연속적인 RPY로 변환 (메인 함수) */
RPY YMatrix::quaternionToRPYSmooth(const Quaternion& q, double sampling_time) {
    if (!integrator_initialized) {
        // 첫 번째 호출: 직접 변환으로 초기화
        last_quaternion = q;
        YMatrix R = fromQuaternion(q.w, q.x, q.y, q.z);
        last_rpy = R.toRPY();
        integrator_initialized = true;
        return last_rpy;
    }

    integration_dt = sampling_time;
    
    // Step 1: 쿼터니언 연속성 보장 - 문제 없어 보임
    Quaternion continuous_q = ensureQuaternionContinuity(q, last_quaternion);
    
    // Step 2: 각속도 계산
    std::vector<double> raw_omega = computeAngularVelocityFromQuaternions(
        last_quaternion, continuous_q, integration_dt);
    
    // Step 3: 필터링 - 문제 없어보임
    std::vector<double> filtered_omega = applyLowPassFilter(
        raw_omega, filtered_angular_velocity, filter_alpha);
    filtered_angular_velocity = filtered_omega;
    
    // Step 4: Euler rate equations로 RPY 업데이트
    std::vector<double> new_rpy = applyEulerRateEquations(last_rpy, filtered_omega, integration_dt);
    
    // 상태 업데이트
    last_quaternion = continuous_q;
    last_rpy = new_rpy;
    
    RPY Reurn_RPY(new_rpy[0], new_rpy[1], new_rpy[2]);
    return Reurn_RPY;
}

/* 쿼터니언 시퀀스를 연속적인 RPY 시퀀스로 변환 */
std::vector<std::vector<double>> YMatrix::quaternionSequenceToRPYSmooth(
    const std::vector<Quaternion>& quat_sequence, double dt) {
    
    if (quat_sequence.empty()) {
        return {};
    }
    
    // 임시로 설정 저장
    double old_dt = integration_dt;
    double old_alpha = filter_alpha;
    bool old_initialized = integrator_initialized;
    
    // 새로운 설정 적용
    setIntegrationParams(dt, filter_alpha);
    resetIntegrator();
    
    std::vector<std::vector<double>> rpy_sequence;
    
    for (const auto& q : quat_sequence) {
        RPY rpy = quaternionToRPYSmooth(q);
        rpy_sequence.push_back({rpy.roll, rpy.pitch, rpy.yaw});
    }
    
    // 원래 설정 복원
    integration_dt = old_dt;
    filter_alpha = old_alpha;
    integrator_initialized = old_initialized;
    
    return rpy_sequence;
}

/* 적분기 설정 */
void YMatrix::setIntegrationParams(double dt, double filter_alpha_param) {
    integration_dt = dt;
    filter_alpha = filter_alpha_param;
}

/* 적분기 리셋 */
void YMatrix::resetIntegrator() {
    integrator_initialized = false;
    filtered_angular_velocity = {0.0, 0.0, 0.0};
    last_rpy = {0.0, 0.0, 0.0};
    last_quaternion = {1.0, 0.0, 0.0, 0.0};
}

/* 수동 적분 (이전 상태를 직접 제공) */
std::vector<double> YMatrix::quaternionToRPYIntegrated(
    const Quaternion& current_q, 
    const Quaternion& previous_q,
    const std::vector<double>& previous_rpy,
    double dt) {
    
    if (previous_rpy.size() != 3) {
        throw std::invalid_argument("Previous RPY must have 3 elements");
    }
    
    // 연속성 보장
    Quaternion continuous_q = ensureQuaternionContinuity(current_q, previous_q);
    
    // 각속도 계산
    std::vector<double> omega = computeAngularVelocityFromQuaternions(previous_q, continuous_q, dt);
    
    // Euler rate equations 적용
    std::vector<double> new_rpy = applyEulerRateEquations(previous_rpy, omega, dt);
    
    return new_rpy;
}

/* 쿼터니언 연속성 보장 */
Quaternion YMatrix::ensureQuaternionContinuity(
    const Quaternion& current, const Quaternion& previous) {
    
    double dot_product = current.w * previous.w + current.x * previous.x + 
                        current.y * previous.y + current.z * previous.z;
    
    if (dot_product < 0.0) {
        return Quaternion(-current.w, -current.x, -current.y, -current.z);
    }
    return current;
}

/* 쿼터니언 차이로부터 각속도 계산 */
std::vector<double> YMatrix::computeAngularVelocityFromQuaternions(
    const Quaternion& q_prev, const Quaternion& q_curr, double dt) {
    
    // q_prev의 켤레
    Quaternion q_prev_conj = {q_prev.w, -q_prev.x, -q_prev.y, -q_prev.z};
    
    // 쿼터니언 차이 계산: dq = q_curr ⊗ q_prev^(-1)
    Quaternion dq;
    dq.w = q_curr.w * q_prev_conj.w - q_curr.x * q_prev_conj.x - 
           q_curr.y * q_prev_conj.y - q_curr.z * q_prev_conj.z;
    dq.x = q_curr.w * q_prev_conj.x + q_curr.x * q_prev_conj.w + 
           q_curr.y * q_prev_conj.z - q_curr.z * q_prev_conj.y;
    dq.y = q_curr.w * q_prev_conj.y - q_curr.x * q_prev_conj.z + 
           q_curr.y * q_prev_conj.w + q_curr.z * q_prev_conj.x;
    dq.z = q_curr.w * q_prev_conj.z + q_curr.x * q_prev_conj.y - 
           q_curr.y * q_prev_conj.x + q_curr.z * q_prev_conj.w;
    
    // 각속도 추출: ω ≈ 2 * [x, y, z] / dt
    return {2.0 * dq.x / dt, 2.0 * dq.y / dt, 2.0 * dq.z / dt};
}

/* Euler Rate Equations 적용 */
std::vector<double> YMatrix::applyEulerRateEquations(
    const std::vector<double>& rpy, const std::vector<double>& omega, double dt) {
    
    if (rpy.size() != 3 || omega.size() != 3) {
        throw std::invalid_argument("RPY and omega must have 3 elements");
    }
    
    double phi = rpy[0];    // roll
    double theta = rpy[1];  // pitch
    double psi = rpy[2];    // yaw
    
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double tan_theta = tan(theta);
    
    double wx = omega[0];
    double wy = omega[1];
    double wz = omega[2];
    
       // Euler rate equations - 수학적으로 항상 적용되어야 함
    double phi_dot, theta_dot, psi_dot;
    
    // Gimbal lock 근처에서의 수치적 처리
    const double gimbal_threshold = 1e-6;  // cos(90°) ≈ 0에 가까울 때만
    
    if (std::abs(cos_theta) < gimbal_threshold) {
        // 정확히 90도나 -90도에서만 특별 처리
        // L'Hôpital의 정리나 극한값 사용
        
        phi_dot = wx;  // 극한값
        theta_dot = wy * cos_phi - wz * sin_phi;  // 여전히 유효
        
        // psi_dot의 극한값 계산
        if (theta > 0) {  // +90도 근처
            psi_dot = (wy * sin_phi + wz * cos_phi) > 0 ? 1e6 : -1e6;
        } else {  // -90도 근처  
            psi_dot = (wy * sin_phi + wz * cos_phi) > 0 ? -1e6 : 1e6;
        }
        
        // 극단적인 값들을 제한
        const double max_extreme_rate = 100.0; // rad/s
        psi_dot = std::max(-max_extreme_rate, std::min(max_extreme_rate, psi_dot));
        
    } else {
        // 일반적인 경우: 정상적인 Euler rate equations
        phi_dot = wx + wy * sin_phi * tan_theta + wz * cos_phi * tan_theta;
        theta_dot = wy * cos_phi - wz * sin_phi;
        psi_dot = (wy * sin_phi + wz * cos_phi) / cos_theta;
    }
    
    // 합리적인 각속도 제한 (물리적 한계)
    const double max_rate = 50.0; // rad/s (약 2865 deg/s)
    phi_dot = std::max(-max_rate, std::min(max_rate, phi_dot));
    theta_dot = std::max(-max_rate, std::min(max_rate, theta_dot));
    psi_dot = std::max(-max_rate, std::min(max_rate, psi_dot));
    
    // 적분으로 새로운 RPY 계산 - 항상 업데이트됨
    std::vector<double> new_rpy = {
        phi + phi_dot * dt,
        theta + theta_dot * dt,
        psi + psi_dot * dt
    };
    
    // 각도 정규화
    new_rpy[0] = normalizeAngle(new_rpy[0]); // roll
    new_rpy[1] = normalizeAngle(new_rpy[1]); // pitch  
    new_rpy[2] = normalizeAngle(new_rpy[2]); // yaw
    
    return new_rpy;
}

/* 로우패스 필터 적용 */
std::vector<double> YMatrix::applyLowPassFilter(
    const std::vector<double>& new_omega, const std::vector<double>& old_omega, double alpha) {
    
    std::vector<double> filtered(3);
    for (int i = 0; i < 3; ++i) {
        filtered[i] = alpha * new_omega[i] + (1.0 - alpha) * old_omega[i];
    }
    return filtered;
}

/****** SpatialAngle Transformation Functions ******/

/* 회전 행렬을 SpatialAngle로 변환 */
SpatialAngle YMatrix::toSpatialAngle() const {
    if (rows() != 3 || cols() != 3) {
        throw std::invalid_argument("Matrix must be 3x3 for spatial angle conversion");
    }
    
    // 회전각도 계산: θ = arccos((trace(R) - 1) / 2)
    double trace = data[0][0] + data[1][1] + data[2][2];
    double angle = acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));

    if (fabs(angle) < 1e-6) {
        // 회전각이 0에 가까운 경우
        return SpatialAngle(0.0, 0.0, 0.0);
    } else if (fabs(angle - M_PI) < 1e-6) {
        // 180도 회전 - 특별한 처리 필요
        double axis_x, axis_y, axis_z;
        
        if (data[0][0] >= data[1][1] && data[0][0] >= data[2][2]) {
            axis_x = sqrt((data[0][0] + 1.0) / 2.0);
            axis_y = data[0][1] / (2.0 * axis_x);
            axis_z = data[0][2] / (2.0 * axis_x);
        } else if (data[1][1] >= data[2][2]) {
            axis_y = sqrt((data[1][1] + 1.0) / 2.0);
            axis_x = data[0][1] / (2.0 * axis_y);
            axis_z = data[1][2] / (2.0 * axis_y);
        } else {
            axis_z = sqrt((data[2][2] + 1.0) / 2.0);
            axis_x = data[0][2] / (2.0 * axis_z);
            axis_y = data[1][2] / (2.0 * axis_z);
        }
        
        return SpatialAngle(axis_x * angle, axis_y * angle, axis_z * angle);
    } else {
        // 일반적인 경우
        double sin_angle = sin(angle);
        double axis_x = (data[2][1] - data[1][2]) / (2.0 * sin_angle);
        double axis_y = (data[0][2] - data[2][0]) / (2.0 * sin_angle);
        double axis_z = (data[1][0] - data[0][1]) / (2.0 * sin_angle);
        
        return SpatialAngle(axis_x * angle, axis_y * angle, axis_z * angle);
    }
}

/* 연속성을 보장하는 SpatialAngle로 변환 */
SpatialAngle YMatrix::toSpatialAngleContinuous(const SpatialAngle& reference) const {
    // 기본 변환
    SpatialAngle base_result = toSpatialAngle();
    
    // 연속성을 보장하는 후보들 생성
    std::vector<SpatialAngle> candidates;
    
    // 1. 기본 결과
    candidates.push_back(base_result);
    
    // 2. 반대 방향 (axis 방향 반전, angle 반전)
    if (base_result.magnitude() > 1e-6) {
        candidates.push_back(SpatialAngle(-base_result.x, -base_result.y, -base_result.z));
    }
    
    // 3. 2π 주기성을 고려한 후보들
    double magnitude = base_result.magnitude();
    if (magnitude > 1e-6) {
        std::vector<double> axis = base_result.axis();
        
        // +2π, -2π 추가
        for (int n = -1; n <= 1; n += 2) {
            double new_angle = magnitude + n * 2.0 * M_PI;
            if (new_angle > 0) {  // 양수 각도만
                candidates.push_back(SpatialAngle(
                    axis[0] * new_angle,
                    axis[1] * new_angle,
                    axis[2] * new_angle
                ));
            }
        }
    }
    
    // 4. 참조값과 가장 가까운 후보 선택
    double min_distance = std::numeric_limits<double>::max();
    SpatialAngle best_candidate = base_result;
    
    for (const auto& candidate : candidates) {
        SpatialAngle diff = candidate - reference;
        double distance = diff.magnitude();
        
        if (distance < min_distance) {
            min_distance = distance;
            best_candidate = candidate;
        }
    }
    
    return best_candidate;
}

/* SpatialAngle로부터 회전 행렬 생성 (수치 안정화) */
YMatrix YMatrix::fromSpatialAngle(const SpatialAngle& spatial_angle) {
    // 회전각도 계산
    double x = spatial_angle.x;
    double y = spatial_angle.y;
    double z = spatial_angle.z;
    double angle = sqrt(x*x + y*y + z*z);
    
    if (angle < 1e-10) {
        return identity(3);
    }
    
    // 회전축 정규화
    double axis_x = x / angle;
    double axis_y = y / angle;
    double axis_z = z / angle;
    
    // Rodrigues' rotation formula
    double c = cos(angle);
    double s = sin(angle);
    double one_minus_c = 1.0 - c;
    
    YMatrix R(3, 3);
    
    R[0][0] = c + axis_x*axis_x*one_minus_c;
    R[0][1] = axis_x*axis_y*one_minus_c - axis_z*s;
    R[0][2] = axis_x*axis_z*one_minus_c + axis_y*s;
    
    R[1][0] = axis_y*axis_x*one_minus_c + axis_z*s;
    R[1][1] = c + axis_y*axis_y*one_minus_c;
    R[1][2] = axis_y*axis_z*one_minus_c - axis_x*s;
    
    R[2][0] = axis_z*axis_x*one_minus_c - axis_y*s;
    R[2][1] = axis_z*axis_y*one_minus_c + axis_x*s;
    R[2][2] = c + axis_z*axis_z*one_minus_c;
    
    // 수치 안정화: Gram-Schmidt 직교화
    return normalizeRotationMatrix(R);
}

/* 연속성을 보장하는 SpatialAngle로부터 회전 행렬 생성 */
YMatrix YMatrix::fromSpatialAngleContinuous(const SpatialAngle& spatial_angle, const YMatrix& reference_R) {
    // 참조 회전행렬의 SpatialAngle 추출
    SpatialAngle reference_spatial = reference_R.toSpatialAngle();
    
    // 연속성을 보장하는 SpatialAngle 선택
    SpatialAngle continuous_spatial = ensureSpatialAngleContinuity(reference_spatial, spatial_angle);
    
    // 안전한 회전행렬 생성
    return fromSpatialAngle(continuous_spatial);
}

/* SpatialAngle 연속성 보장 (정적 헬퍼 함수) */
SpatialAngle YMatrix::ensureSpatialAngleContinuity(const SpatialAngle& reference, const SpatialAngle& target) {
    std::vector<SpatialAngle> candidates;
    
    // 1. 기본 목표값
    candidates.push_back(target);
    
    // 2. 반대 방향
    candidates.push_back(SpatialAngle(-target.x, -target.y, -target.z));
    
    // 3. 2π 주기성 고려
    double magnitude = target.magnitude();
    if (magnitude > 1e-6) {
        std::vector<double> axis = target.axis();
        
        // ±2π 추가
        for (int n = -1; n <= 1; n += 2) {
            double new_angle = magnitude + n * 2.0 * M_PI;
            if (new_angle > 0) {
                candidates.push_back(SpatialAngle(
                    axis[0] * new_angle,
                    axis[1] * new_angle,
                    axis[2] * new_angle
                ));
                
                // 반대 방향도 추가
                candidates.push_back(SpatialAngle(
                    -axis[0] * new_angle,
                    -axis[1] * new_angle,
                    -axis[2] * new_angle
                ));
            }
        }
    }
    
    // 4. 가장 가까운 후보 선택
    double min_distance = std::numeric_limits<double>::max();
    SpatialAngle best = target;
    
    for (const auto& candidate : candidates) {
        SpatialAngle diff = candidate - reference;
        double distance = diff.magnitude();
        
        if (distance < min_distance) {
            min_distance = distance;
            best = candidate;
        }
    }
    
    return best;
}

/* 회전행렬 정규화 (Gram-Schmidt) */
YMatrix YMatrix::normalizeRotationMatrix(const YMatrix& R) {
    if (R.rows() != 3 || R.cols() != 3) {
        return R;
    }
    
    YMatrix R_norm(3, 3);
    
    // 첫 번째 열벡터 정규화
    double norm1 = sqrt(R[0][0]*R[0][0] + R[1][0]*R[1][0] + R[2][0]*R[2][0]);
    if (norm1 < 1e-10) norm1 = 1.0;
    
    R_norm[0][0] = R[0][0] / norm1;
    R_norm[1][0] = R[1][0] / norm1;
    R_norm[2][0] = R[2][0] / norm1;
    
    // 두 번째 열벡터 직교화 및 정규화
    double dot = R_norm[0][0]*R[0][1] + R_norm[1][0]*R[1][1] + R_norm[2][0]*R[2][1];
    double v2_x = R[0][1] - dot * R_norm[0][0];
    double v2_y = R[1][1] - dot * R_norm[1][0];
    double v2_z = R[2][1] - dot * R_norm[2][0];
    
    double norm2 = sqrt(v2_x*v2_x + v2_y*v2_y + v2_z*v2_z);
    if (norm2 < 1e-10) norm2 = 1.0;
    
    R_norm[0][1] = v2_x / norm2;
    R_norm[1][1] = v2_y / norm2;
    R_norm[2][1] = v2_z / norm2;
    
    // 세 번째 열벡터는 외적으로 계산 (완벽한 직교성)
    R_norm[0][2] = R_norm[1][0] * R_norm[2][1] - R_norm[2][0] * R_norm[1][1];
    R_norm[1][2] = R_norm[2][0] * R_norm[0][1] - R_norm[0][0] * R_norm[2][1];
    R_norm[2][2] = R_norm[0][0] * R_norm[1][1] - R_norm[1][0] * R_norm[0][1];
    
    return R_norm;
}


/* SpatialAngle을 Quaternion으로 변환 */
Quaternion YMatrix::spatialAngleToQuaternion(const SpatialAngle& spatial_angle) {
    double angle = spatial_angle.magnitude();
    
    if (angle < 1e-10) {
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }
    
    // 회전축 정규화
    std::vector<double> axis = spatial_angle.axis();
    
    // 쿼터니언 변환: q = [cos(θ/2), sin(θ/2)*axis]
    double half_angle = angle * 0.5;
    double sin_half = sin(half_angle);
    double cos_half = cos(half_angle);
    
    return Quaternion(
        cos_half,
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half
    );
}

/* Quaternion을 SpatialAngle로 변환 */
SpatialAngle YMatrix::quaternionToSpatialAngle(const Quaternion& q) {
    // 쿼터니언 정규화
    double norm = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (norm < 1e-10) {
        printf("Warning: Quaternion to SpatialAngle conversion resulted in near-zero norm(1).\n");
        return SpatialAngle(0.0, 0.0, 0.0);
    }
    
    double w = q.w / norm;
    double x = q.x / norm;
    double y = q.y / norm;
    double z = q.z / norm;
    
    // w의 부호 조정 (shortest path 선택)
    if (w < 0) {
        w = -w; x = -x; y = -y; z = -z;
    }
    
    // 회전각도 계산: θ = 2 * arccos(|w|)
    double angle = 2.0 * acos(std::min(1.0, fabs(w)));
    
    if (angle < 1e-6) {
        // printf("Warning: Conversion resulted in near-zero angle(2) - %.3f.\n",angle);
        return SpatialAngle(0.0, 0.0, 0.0);
    }
    
    // 회전축 추출: axis = [x, y, z] / sin(θ/2)
    double sin_half_angle = sin(angle * 0.5);
    
    if (fabs(sin_half_angle) < 1e-10) {
        // printf("Warning: Conversion resulted in near-zero axis(3) - %.3f.\n",sin_half_angle);
        return SpatialAngle(0.0, 0.0, 0.0);
    }
    
    double axis_x = x / sin_half_angle;
    double axis_y = y / sin_half_angle;
    double axis_z = z / sin_half_angle;
    
    // SpatialAngle = axis * angle
    return SpatialAngle(axis_x * angle, axis_y * angle, axis_z * angle);
}

/* 두 회전 행렬 사이의 SpatialAngle 오차 계산 */
SpatialAngle YMatrix::spatialAngularError(const YMatrix& target_R, const YMatrix& current_R) {
    if (target_R.rows() != 3 || target_R.cols() != 3 || 
        current_R.rows() != 3 || current_R.cols() != 3) {
        throw std::invalid_argument("Both matrices must be 3x3 rotation matrices");
    }
    
    // R_err = R_target * R_current^T
    YMatrix R_err = target_R * current_R.transpose();
    
    // skew-symmetric 행렬에서 벡터 추출
    double e_x = 0.5 * (R_err[2][1] - R_err[1][2]);
    double e_y = 0.5 * (R_err[0][2] - R_err[2][0]);
    double e_z = 0.5 * (R_err[1][0] - R_err[0][1]);
    
    return SpatialAngle(e_x, e_y, e_z);
}

/* SpatialAngle 보간 (SLERP 기반) */
SpatialAngle YMatrix::interpolateSpatialAngle(
    const SpatialAngle& spatial_angle1,
    const SpatialAngle& spatial_angle2,
    double t) {
    
    if (t <= 0.0) return spatial_angle1;
    if (t >= 1.0) return spatial_angle2;
    
    // 쿼터니언으로 변환하여 SLERP 수행
    Quaternion q1 = spatialAngleToQuaternion(spatial_angle1);
    Quaternion q2 = spatialAngleToQuaternion(spatial_angle2);
    
    // 내적 계산
    double dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
    
    // 최단 경로 선택
    if (dot < 0.0) {
        q2.w = -q2.w; q2.x = -q2.x; q2.y = -q2.y; q2.z = -q2.z;
        dot = -dot;
    }
    
    Quaternion q_result;
    
    if (dot > 0.9995) {
        // 선형 보간 (너무 가까운 경우)
        q_result.w = q1.w + t * (q2.w - q1.w);
        q_result.x = q1.x + t * (q2.x - q1.x);
        q_result.y = q1.y + t * (q2.y - q1.y);
        q_result.z = q1.z + t * (q2.z - q1.z);
    } else {
        // 구면 보간
        double theta_0 = acos(abs(dot));
        double sin_theta_0 = sin(theta_0);
        double theta = theta_0 * t;
        double sin_theta = sin(theta);
        
        double s0 = cos(theta) - dot * sin_theta / sin_theta_0;
        double s1 = sin_theta / sin_theta_0;
        
        q_result.w = s0 * q1.w + s1 * q2.w;
        q_result.x = s0 * q1.x + s1 * q2.x;
        q_result.y = s0 * q1.y + s1 * q2.y;
        q_result.z = s0 * q1.z + s1 * q2.z;
    }
    
    // 정규화
    double norm = sqrt(q_result.w*q_result.w + q_result.x*q_result.x + 
                        q_result.y*q_result.y + q_result.z*q_result.z);
    if (norm > 1e-10) {
        q_result.w /= norm; q_result.x /= norm; 
        q_result.y /= norm; q_result.z /= norm;
    }
    
    return quaternionToSpatialAngle(q_result);
}

/* RPY에서 SpatialAngle로 변환 */
SpatialAngle YMatrix::rpyToSpatialAngle(const RPY& rpy) {
    YMatrix R = fromRPY(rpy.roll, rpy.pitch, rpy.yaw);
    return R.toSpatialAngle();
}

/* SpatialAngle에서 RPY로 변환 */
RPY YMatrix::spatialAngleToRPY(const SpatialAngle& spatial_angle) {
    YMatrix R = fromSpatialAngle(spatial_angle);
    std::vector<double> rpy = R.toRPY();
    return RPY(rpy[0], rpy[1], rpy[2]);
}