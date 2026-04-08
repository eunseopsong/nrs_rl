#include "Y2Kinematics/QP_solver.hpp"

// ---------- CSC conversion: upper-triangular for P ----------
QPSolver::CSCMatrix QPSolver::convertToCSC_upperTri(const YMatrix& matrix, double tolerance) {
    CSCMatrix csc;
    csc.nrows = static_cast<OSQPInt>(matrix.rows());
    csc.ncols = static_cast<OSQPInt>(matrix.cols());
    csc.nnz   = 0;

    for (size_t j = 0; j < matrix.cols(); ++j) {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            if (i > j) continue; // upper-triangular
            if (std::abs(matrix[i][j]) > tolerance) csc.nnz++;
        }
    }

    csc.data.reserve(csc.nnz);
    csc.indices.reserve(csc.nnz);
    csc.indptr.reserve(csc.ncols + 1);

    csc.indptr.push_back(0);
    for (size_t j = 0; j < matrix.cols(); ++j) {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            if (i > j) continue;
            if (std::abs(matrix[i][j]) > tolerance) {
                csc.data.push_back(static_cast<OSQPFloat>(matrix[i][j]));
                csc.indices.push_back(static_cast<OSQPInt>(i));
            }
        }
        csc.indptr.push_back(static_cast<OSQPInt>(csc.data.size()));
    }

    return csc;
}

// ---------- CSC conversion: full for A ----------
QPSolver::CSCMatrix QPSolver::convertToCSC_full(const YMatrix& matrix, double tolerance) {
    CSCMatrix csc;
    csc.nrows = static_cast<OSQPInt>(matrix.rows());
    csc.ncols = static_cast<OSQPInt>(matrix.cols());
    csc.nnz   = 0;

    for (size_t j = 0; j < matrix.cols(); ++j) {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            if (std::abs(matrix[i][j]) > tolerance) csc.nnz++;
        }
    }

    csc.data.reserve(csc.nnz);
    csc.indices.reserve(csc.nnz);
    csc.indptr.reserve(csc.ncols + 1);

    csc.indptr.push_back(0);
    for (size_t j = 0; j < matrix.cols(); ++j) {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            if (std::abs(matrix[i][j]) > tolerance) {
                csc.data.push_back(static_cast<OSQPFloat>(matrix[i][j]));
                csc.indices.push_back(static_cast<OSQPInt>(i));
            }
        }
        csc.indptr.push_back(static_cast<OSQPInt>(csc.data.size()));
    }

    return csc;
}

// ---------- Solve QP using OSQP (warm-start included) ----------
QPSolver::QPResult QPSolver::solve(const YMatrix& H, const std::vector<double>& f,
                                   const YMatrix& A_eq, const std::vector<double>& b_eq,
                                   const std::vector<double>& lb, const std::vector<double>& ub,
                                   double tolerance) {
    QPResult result;
    result.success = false;
    result.exitflag = 0;

    // ---------- Input validation ----------
    if (H.rows() != H.cols()) {
        std::cerr << "[QPSolver] Error: H matrix must be square\n";
        return result;
    }
    if (f.size() != static_cast<size_t>(H.rows())) {
        std::cerr << "[QPSolver] Error: f vector size mismatch with H\n";
        return result;
    }
    if (lb.size() != f.size() || ub.size() != f.size()) {
        std::cerr << "[QPSolver] Error: bound vector size mismatch\n";
        return result;
    }
    if (A_eq.cols() != H.rows()) {
        std::cerr << "[QPSolver] Error: A_eq column size mismatch with H\n";
        return result;
    }
    if (b_eq.size() != static_cast<size_t>(A_eq.rows())) {
        std::cerr << "[QPSolver] Error: b_eq size mismatch with A_eq rows\n";
        return result;
    }

    const OSQPInt n    = static_cast<OSQPInt>(H.rows());
    const OSQPInt m_eq = static_cast<OSQPInt>(A_eq.rows());
    const OSQPInt m    = m_eq + n; // A = [Aeq; I], l=[beq; lb], u=[beq; ub]

    // ---------- Symmetrize H (IMPORTANT for OSQP) ----------
    YMatrix Hsym = H;
    for (OSQPInt i = 0; i < n; ++i) {
        for (OSQPInt j = i + 1; j < n; ++j) {
            const double v = 0.5 * (H[i][j] + H[j][i]);
            Hsym[i][j] = v;
            Hsym[j][i] = v;
        }
    }

    // ---------- P (upper-triangular) ----------
    CSCMatrix P_csc = convertToCSC_upperTri(Hsym);
    if (P_csc.nnz == 0) {
        std::cerr << "[QPSolver] Warning: P is all zeros; adding tiny diagonal regularization\n";
        YMatrix Hreg = Hsym;
        for (OSQPInt i = 0; i < n; ++i) Hreg[i][i] += 1e-8;
        P_csc = convertToCSC_upperTri(Hreg);
    }
    if (P_csc.data.empty()) {
        std::cerr << "[QPSolver] Error: empty P CSC\n";
        return result;
    }

    // ---------- Build A_full and l/u ----------
    YMatrix A_full(m, n);

    // 반드시 0 초기화 (YMatrix가 기본 초기화를 보장하지 않으면 간헐 실패 원인)
    for (OSQPInt i = 0; i < m; ++i)
        for (OSQPInt j = 0; j < n; ++j)
            A_full[i][j] = 0.0;

    std::vector<OSQPFloat> l_full(m), u_full(m);

    // equalities: Aeq x = beq  => l=u=beq
    for (OSQPInt i = 0; i < m_eq; ++i) {
        for (OSQPInt j = 0; j < n; ++j) A_full[i][j] = A_eq[i][j];
        l_full[i] = static_cast<OSQPFloat>(b_eq[i]);
        u_full[i] = static_cast<OSQPFloat>(b_eq[i]);
    }

    // bounds: I x in [lb, ub]
    for (OSQPInt i = 0; i < n; ++i) {
        A_full[m_eq + i][i] = 1.0;
        l_full[m_eq + i] = static_cast<OSQPFloat>(lb[i]);
        u_full[m_eq + i] = static_cast<OSQPFloat>(ub[i]);
    }

    CSCMatrix A_csc = convertToCSC_full(A_full);
    if (A_csc.data.empty()) {
        std::cerr << "[QPSolver] Error: empty A CSC\n";
        return result;
    }

    // ---------- q ----------
    std::vector<OSQPFloat> q_data(f.size());
    for (size_t i = 0; i < f.size(); ++i) q_data[i] = static_cast<OSQPFloat>(f[i]);

    // ---------- Create OSQP CSC matrices ----------
    OSQPCscMatrix* P = OSQPCscMatrix_new(P_csc.nrows, P_csc.ncols, P_csc.nnz,
                                         P_csc.data.data(), P_csc.indices.data(), P_csc.indptr.data());
    OSQPCscMatrix* A = OSQPCscMatrix_new(A_csc.nrows, A_csc.ncols, A_csc.nnz,
                                         A_csc.data.data(), A_csc.indices.data(), A_csc.indptr.data());
    if (!P || !A) {
        std::cerr << "[QPSolver] Error: failed to create CSC matrices\n";
        if (P) OSQPCscMatrix_free(P);
        if (A) OSQPCscMatrix_free(A);
        return result;
    }

    // ---------- Settings (MINIMAL, version-safe) ----------
    OSQPSettings* settings = OSQPSettings_new();
    if (!settings) {
        std::cerr << "[QPSolver] Error: failed to create settings\n";
        OSQPCscMatrix_free(P);
        OSQPCscMatrix_free(A);
        return result;
    }

    settings->eps_abs  = static_cast<OSQPFloat>(tolerance);
    settings->eps_rel  = static_cast<OSQPFloat>(tolerance);
    settings->verbose  = 0;
    settings->max_iter = 20000;

    // ⚠️ 아래는 버전별로 필드가 없을 수 있어 사용하지 않음
    // settings->polish = 0;
    // settings->adaptive_rho = 1;
    // settings->scaled_termination = 1;
    // settings->check_termination = 25;

    // ---------- Setup ----------
    OSQPSolver* solver = nullptr;
    OSQPInt exitflag = osqp_setup(&solver, P, q_data.data(), A,
                                  l_full.data(), u_full.data(), m, n, settings);

    if (exitflag != 0 || !solver) {
        std::cerr << "[QPSolver] Error: osqp_setup failed, exitflag=" << exitflag << "\n";
        result.exitflag = exitflag;
        OSQPCscMatrix_free(P);
        OSQPCscMatrix_free(A);
        OSQPSettings_free(settings);
        return result;
    }

    // ---------- Warm-start ----------
    if (has_warm_ && last_n_ == n && last_m_ == m) {
        const OSQPFloat* x0 = warm_x_.empty() ? nullptr : warm_x_.data();
        const OSQPFloat* y0 = (warm_y_.size() == static_cast<size_t>(m)) ? warm_y_.data() : nullptr;
        (void)osqp_warm_start(solver, x0, y0);
    } else {
        clearWarmStart();
    }

    // ---------- Solve ----------
    exitflag = osqp_solve(solver);
    result.exitflag = exitflag;

    // Fill diagnostics (even on failure)
    if (solver->info) {
        result.status = solver->info->status_val;
        result.status_string = solver->info->status ? solver->info->status : "";
        result.iter = solver->info->iter;
        result.objective_value = solver->info->obj_val;
    } else {
        result.status = 0;
        result.status_string = "no solver->info";
        result.iter = 0;
        result.objective_value = 0.0;
    }

    // Decide success by OSQP status (NOT by exitflag)
    result.success = (result.status == OSQP_SOLVED || result.status == OSQP_SOLVED_INACCURATE);

    // Copy solution if available
    if (solver->solution && solver->solution->x) {
        result.solution.resize(n);
        for (OSQPInt i = 0; i < n; ++i) {
            result.solution[i] = static_cast<double>(solver->solution->x[i]);
        }
    }

    // Cache warm-start only if solved
    if (result.success && solver->solution && solver->solution->x) {
        warm_x_.assign(solver->solution->x, solver->solution->x + n);
        if (solver->solution->y) warm_y_.assign(solver->solution->y, solver->solution->y + m);
        else warm_y_.assign(static_cast<size_t>(m), static_cast<OSQPFloat>(0.0));
        has_warm_ = true;
        last_n_ = n;
        last_m_ = m;
    } else {
        clearWarmStart();
    }

    // ---------- Cleanup ----------
    osqp_cleanup(solver);
    OSQPCscMatrix_free(P);
    OSQPCscMatrix_free(A);
    OSQPSettings_free(settings);

    return result;
}
