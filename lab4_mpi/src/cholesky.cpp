#include "cholesky.hpp"

#include <algorithm>
#include <cmath>

namespace lab4 {

namespace {

// Скалярное произведение векторов длины len — основная операция варианта.
double dot(const double* x, const double* y, int len) {
    double s = 0.0;
    for (int k = 0; k < len; ++k) s += x[k] * y[k];
    return s;
}

double* row(double* m, int ld, int i) { return m + static_cast<std::size_t>(i) * ld; }
const double* row(const double* m, int ld, int i) { return m + static_cast<std::size_t>(i) * ld; }

double checked_sqrt(double d) {
    if (!(d > 0.0)) throw NotPositiveDefinite("Матрица не является положительно определённой");
    return std::sqrt(d);
}

}  // namespace

double test_matrix_entry(int i, int j) {
    if (i == j) return 4.0 + i / 9.0;
    return 2.0 / 3.0 + std::min(i, j) / 9.0;
}

double exact_factor_entry(int i, int j) {
    if (i == j) return 2.0;
    return i > j ? 1.0 / 3.0 : 0.0;
}

void fill_test_rows(double* a, int ld, int i0, int rows, int n) {
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j) row(a, ld, r)[j] = test_matrix_entry(i0 + r, j);
}

std::vector<double> make_test_matrix(int n) {
    std::vector<double> a(static_cast<std::size_t>(n) * n);
    fill_test_rows(a.data(), n, 0, n, n);
    return a;
}

void factor_diagonal_block(double* a, int ld, int m) {
    for (int i = 0; i < m; ++i) {
        double* ai = row(a, ld, i);
        for (int j = 0; j < i; ++j) {
            const double* aj = row(a, ld, j);
            ai[j] = (ai[j] - dot(ai, aj, j)) / aj[j];
        }
        ai[i] = checked_sqrt(ai[i] - dot(ai, ai, i));
    }
}

void solve_lower_transposed(double* x, int ld_x, int rows, int m, const double* l, int ld_l) {
    // Строка r матрицы X: x_t = (b_t − Σ_{k<t} x_k·l_tk) / l_tt — прямая подстановка.
    for (int r = 0; r < rows; ++r) {
        double* xr = row(x, ld_x, r);
        for (int t = 0; t < m; ++t) {
            const double* lt = row(l, ld_l, t);
            xr[t] = (xr[t] - dot(xr, lt, t)) / lt[t];
        }
    }
}

void modify_block(const double* a_i, double* l_i, int ld, const double* l_j, int ld_j, int n, int b, int ib,
                  int jb) {
    const int mi = std::min(b, n - ib * b), j0 = jb * b, mj = std::min(b, n - j0);
    const bool diagonal = ib == jb;
    for (int r = 0; r < mi; ++r) {
        const int cols = diagonal ? r + 1 : mj;
        for (int c = 0; c < cols; ++c) row(l_i, ld, r)[j0 + c] = row(a_i, ld, r)[j0 + c];
    }
    for (int kb = 0; kb < jb; ++kb) {  // блоки k < j всегда полные (b×b)
        const int k0 = kb * b;
        for (int r = 0; r < mi; ++r) {
            const double* lik = row(l_i, ld, r) + k0;
            const int cols = diagonal ? r + 1 : mj;
            for (int c = 0; c < cols; ++c) row(l_i, ld, r)[j0 + c] -= dot(lik, row(l_j, ld_j, c) + k0, b);
        }
    }
}

void factor_block(double* l_j, int ld, int n, int b, int jb) {
    const int j0 = jb * b;
    factor_diagonal_block(l_j + j0, ld, std::min(b, n - j0));
}

void solve_block(double* l_i, int ld, const double* l_jj, int ld_jj, int n, int b, int ib, int jb) {
    const int j0 = jb * b;
    solve_lower_transposed(l_i + j0, ld, std::min(b, n - ib * b), std::min(b, n - j0), l_jj, ld_jj);
}

void pack_row(const double* l_j, int ld, int mj, int j0, double* out) {
    for (int r = 0; r < mj; ++r) std::copy(row(l_j, ld, r), row(l_j, ld, r) + j0, out + static_cast<std::size_t>(r) * j0);
    double* diag = out + static_cast<std::size_t>(mj) * j0;
    for (int r = 0; r < mj; ++r) std::copy(row(l_j, ld, r) + j0, row(l_j, ld, r) + j0 + mj, diag + r * mj);
}

void cholesky_block(const double* a, double* l, int n, int b) {
    const int p = block_count(n, b);
    for (int jb = 0; jb < p; ++jb) {  // блочные столбцы — последовательно: столбец j нужен всем следующим
        const int j0 = jb * b;
        double* lj = row(l, n, j0);
        modify_block(row(a, n, j0), lj, n, lj, n, n, b, jb, jb);
        factor_block(lj, n, n, b, jb);
        for (int ib = jb + 1; ib < p; ++ib) {
            const int i0 = ib * b;
            modify_block(row(a, n, i0), row(l, n, i0), n, lj, n, n, b, ib, jb);
            solve_block(row(l, n, i0), n, lj + j0, n, n, b, ib, jb);
        }
    }
}

double max_error_vs_exact(const double* l, int ld, int i0, int rows, int n) {
    double err = 0.0;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j)
            err = std::max(err, std::abs(row(l, ld, r)[j] - exact_factor_entry(i0 + r, j)));
    return err;
}

}  // namespace lab4
