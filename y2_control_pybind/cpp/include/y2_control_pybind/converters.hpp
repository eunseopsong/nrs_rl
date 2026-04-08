#pragma once

#include "Y2Matrix/YMatrix.hpp"

#include <stdexcept>
#include <vector>

namespace y2_control_pybind {

inline YMatrix vector2dToYMatrix(const std::vector<std::vector<double>>& data) {
    if (data.empty()) {
        throw std::invalid_argument("vector2dToYMatrix: input is empty");
    }

    const size_t rows = data.size();
    const size_t cols = data[0].size();

    if (cols == 0) {
        throw std::invalid_argument("vector2dToYMatrix: input has zero columns");
    }

    for (size_t i = 1; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::invalid_argument("vector2dToYMatrix: ragged 2D vector is not allowed");
        }
    }

    YMatrix mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = data[i][j];
        }
    }
    return mat;
}

inline std::vector<std::vector<double>> yMatrixToVector2d(const YMatrix& mat) {
    std::vector<std::vector<double>> out(mat.rows(), std::vector<double>(mat.cols(), 0.0));
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            out[i][j] = mat[i][j];
        }
    }
    return out;
}

inline void validateJointVector(const std::vector<double>& q, size_t expected_size, const char* name) {
    if (q.size() != expected_size) {
        throw std::invalid_argument(std::string(name) + ": expected " +
                                    std::to_string(expected_size) + " elements, got " +
                                    std::to_string(q.size()));
    }
}

inline void validateHTM4x4(const std::vector<std::vector<double>>& htm, const char* name) {
    if (htm.size() != 4) {
        throw std::invalid_argument(std::string(name) + ": expected 4 rows");
    }
    for (size_t i = 0; i < 4; ++i) {
        if (htm[i].size() != 4) {
            throw std::invalid_argument(std::string(name) + ": expected 4 columns in every row");
        }
    }
}

}  // namespace y2_control_pybind