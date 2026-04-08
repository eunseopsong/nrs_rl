#pragma once
#include <osqp/osqp.h>
#include "Y2Matrix/YMatrix.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

class QPSolver {
private:
    struct CSCMatrix {
        std::vector<OSQPFloat> data;
        std::vector<OSQPInt>   indices;
        std::vector<OSQPInt>   indptr;
        OSQPInt nrows = 0, ncols = 0, nnz = 0;
    };

    // P: upper-triangular only (OSQP requirement)
    CSCMatrix convertToCSC_upperTri(const YMatrix& matrix, double tolerance = 1e-12);

    // A: full
    CSCMatrix convertToCSC_full(const YMatrix& matrix, double tolerance = 1e-12);

    // Warm-start buffers
    std::vector<OSQPFloat> warm_x_;   // length n
    std::vector<OSQPFloat> warm_y_;   // length m
    bool    has_warm_ = false;
    OSQPInt last_n_   = 0;
    OSQPInt last_m_   = 0;

public:
    struct QPResult {
        std::vector<double> solution;
        bool success = false;

        // OSQP status info
        OSQPInt status = 0;               // solver->info->status_val
        std::string status_string;        // solver->info->status
        OSQPInt iter = 0;                 // solver->info->iter
        double objective_value = 0.0;     // solver->info->obj_val

        // API-level return code (debug only)
        OSQPInt exitflag = 0;             // return of osqp_setup / osqp_solve
    };

    QPSolver() = default;

    void setWarmStart(const std::vector<double>& x0,
                      const std::vector<double>& y0 = std::vector<double>()) {
        warm_x_.assign(x0.begin(), x0.end());
        warm_y_.assign(y0.begin(), y0.end());
        has_warm_ = !warm_x_.empty();
        last_n_ = static_cast<OSQPInt>(warm_x_.size());
        last_m_ = static_cast<OSQPInt>(warm_y_.size());
    }

    void clearWarmStart() {
        warm_x_.clear();
        warm_y_.clear();
        has_warm_ = false;
        last_n_ = 0;
        last_m_ = 0;
    }

    // Solve:
    //   min  1/2 x' H x + f' x
    //   s.t. A_eq x = b_eq
    //        lb <= x <= ub
    QPResult solve(const YMatrix& H, const std::vector<double>& f,
                   const YMatrix& A_eq, const std::vector<double>& b_eq,
                   const std::vector<double>& lb, const std::vector<double>& ub,
                   double tolerance = 1e-4);
};
