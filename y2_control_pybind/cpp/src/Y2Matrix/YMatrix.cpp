#include "Y2Matrix/YMatrix.hpp"

#include <sstream>   // 추가: std::istringstream

/* 행, 열 크기를 설정하지 않은 경우 */
YMatrix::YMatrix(std::initializer_list<std::initializer_list<double>> list) {
    for (const auto& row : list) {
        data.emplace_back(row);     // 이 row를 data 벡터에 추가
    }
}
    
/* 행렬 덧셈 */
YMatrix YMatrix::operator+(const YMatrix& other) const {
    YMatrix result(rows(), cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result[i][j] = data[i][j] + other[i][j];
        }
    }
    return result;
}

/* 행렬 뺄셈 🔹 추가 */
YMatrix YMatrix::operator-(const YMatrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    YMatrix result(rows(), cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result[i][j] = data[i][j] - other[i][j];
        }
    }
    return result;
}
    
/* 행렬 곱셈 */
YMatrix YMatrix::operator*(const YMatrix& other) const {
    YMatrix result(rows(), other.cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            for (size_t k = 0; k < cols(); ++k) {
                result[i][j] += data[i][k] * other[k][j];
            }
        }
    }
    return result;
}

/* 스칼라 곱셈 연산자 (행렬 * 스칼라) */
YMatrix YMatrix::operator*(double scalar) const {
    YMatrix result(rows(), cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

/* 스칼라 곱셈 연산자 (스칼라 * 행렬) */
inline YMatrix operator*(double scalar, const YMatrix& matrix) {
    return matrix * scalar;
}

/* 스칼라 나눗셈 연산자 */
YMatrix YMatrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero");
    }
    YMatrix result(rows(), cols());
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result[i][j] = data[i][j] / scalar;
        }
    }
    return result;
}

/* 전치행렬 */
YMatrix YMatrix::transpose() const {
    YMatrix result(cols(), rows());  // 행과 열 크기 바뀜
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result[j][i] = data[i][j];  // 행-열 인덱스 바뀜
        }
    }
    return result;
}

/* 역행렬 계산 (가우스-조던 소거법 사용) */
YMatrix YMatrix::inverse() const {
    if (rows() != cols()) {
        throw std::invalid_argument("Inverse can only be calculated for square matrices");
    }
    
    size_t n = rows();
    const double tolerance = 1e-12;
    
    // 확장 행렬 [A|I] 생성
    YMatrix augmented(n, 2 * n);
    
    // 원본 행렬을 왼쪽에 복사
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented[i][j] = data[i][j];
        }
    }
    
    // 단위행렬을 오른쪽에 생성
    for (size_t i = 0; i < n; ++i) {
        augmented[i][i + n] = 1.0;
    }
    
    // 가우스-조던 소거법 수행
    for (size_t i = 0; i < n; ++i) {
        // 피벗 찾기 (부분 피벗팅)
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        // 피벗이 0에 가까우면 특이행렬
        if (std::abs(augmented[maxRow][i]) < tolerance) {
            throw std::runtime_error("Matrix is singular (non-invertible)");
        }
        
        // 행 교환
        if (maxRow != i) {
            for (size_t j = 0; j < 2 * n; ++j) {
                std::swap(augmented[i][j], augmented[maxRow][j]);
            }
        }
        
        // 피벗으로 행 정규화
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * n; ++j) {
            augmented[i][j] /= pivot;
        }
        
        // 다른 행들에서 현재 열을 소거
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented[k][i];
                for (size_t j = 0; j < 2 * n; ++j) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // 역행렬 추출 (오른쪽 절반)
    YMatrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i][j] = augmented[i][j + n];
        }
    }
    
    return result;
}

/* Extraction function */
YMatrix YMatrix::extract(size_t start_row, size_t start_col,
                         size_t num_rows, size_t num_cols) const {
    if (start_row + num_rows > rows() || start_col + num_cols > cols()) {
        throw std::out_of_range("Extract range exceeds matrix dimensions.");
    }
    YMatrix result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            result[i][j] = data[start_row + i][start_col + j];
        }
    }
    return result;
}

/* Insert function */
void YMatrix::insert(size_t start_row, size_t start_col, const YMatrix& others) {
    if (start_row + others.rows() > rows() || start_col + others.cols() > cols()) {
        throw std::out_of_range("Insert range exceeds matrix dimensions.");
    }
    for (size_t i = 0; i < others.rows(); ++i) {
        for (size_t j = 0; j < others.cols(); ++j) {
            data[start_row + i][start_col + j] = others[i][j];  // others의 값을 data에 삽입
        }
    }
}

/* 행렬 크기 조정 (기존 데이터 보존) */
void YMatrix::resize(size_t new_rows, size_t new_cols) {
    // 새로운 크기의 행렬 생성 (0으로 초기화)
    std::vector<std::vector<double>> new_data(new_rows, std::vector<double>(new_cols, 0.0));
    
    // 기존 데이터를 새로운 행렬에 복사
    size_t copy_rows = std::min(rows(), new_rows);
    size_t copy_cols = std::min(cols(), new_cols);
    
    for (size_t i = 0; i < copy_rows; ++i) {
        for (size_t j = 0; j < copy_cols; ++j) {
            new_data[i][j] = data[i][j];
        }
    }
    
    data = std::move(new_data);
}

/* Matrix to std::vector */
std::vector<double> YMatrix::toVector() const {
    std::vector<double> vec;
    for (const auto& row : data) {
        vec.insert(vec.end(), row.begin(), row.end());
    }
    return vec;
}

/* Vertical append function */
void YMatrix::appendV(const YMatrix& object){
    for (size_t i = 0; i < object.rows(); ++i) {
        if (object.cols() != cols()) {
            throw std::invalid_argument("Number of col is not correct.");
        }
        data.push_back(object[i]);  // VMatrix의 각 행을 YMatrix에 추가
    }
}

/* Horizontal append function */
void YMatrix::appendH(const YMatrix& object) {
    if (object.rows() != rows()) {
        throw std::invalid_argument("Number of row is not correct.");
    }
    for (size_t i = 0; i < rows(); ++i) {
        data[i].insert(data[i].end(), object[i].begin(), object[i].end());  // 각 행에 object의 행을 추가
    }
}
    
/* 출력 */
void YMatrix::print() const {
    for (const auto& row : data) {
        for (const double& val : row) {
            std::cout << std::setw(8) << val << " ";
        }
        std::cout << std::endl;
    }
}

/* 행렬을 txt 파일로 저장 */
void YMatrix::saveToFile(const std::string& filePath, int precision) const {
    std::ofstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filePath);
    }
    
    file << std::fixed << std::setprecision(precision);
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            file << data[i][j];
            if (j < cols() - 1) {
                file << " ";
            }
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Matrix saved successfully to: " << filePath << std::endl;
}

/* txt 파일로부터 행렬 로드 */
YMatrix YMatrix::loadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filePath);
    }
    
    std::vector<std::vector<double>> matrix_;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) row.push_back(val);
        if (!row.empty()) matrix_.push_back(row);
    }

    size_t rows = matrix_.size();
    size_t cols = matrix_.empty() ? 0 : matrix_[0].size();
    
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Matrix is empty or malformed in file: " + filePath);
    } else {
        std::cout << "Matrix loaded successfully from: " << filePath << std::endl;
        printf("Matrix size: %zu rows, %zu cols\n", rows, cols);
    }

    YMatrix matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = matrix_[i][j];
        }
    }
    
    file.close();
    std::cout << "Matrix loaded successfully from: " << filePath << std::endl;
    return matrix;
}
