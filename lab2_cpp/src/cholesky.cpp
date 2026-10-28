#include "cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace lab2 {

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

std::vector<double> make_test_matrix(int n) {
    std::vector<double> a(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) row(a.data(), n, i)[j] = test_matrix_entry(i, j);
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

void cholesky_point(const double* a, double* l, int n) {
    for (int j = 0; j < n; ++j) {
        double* lj = row(l, n, j);
        lj[j] = checked_sqrt(row(a, n, j)[j] - dot(lj, lj, j));
        for (int i = j + 1; i < n; ++i) {
            double* li = row(l, n, i);
            li[j] = (row(a, n, i)[j] - dot(li, lj, j)) / lj[j];
        }
    }
}

void cholesky_block(const double* a, double* l, int n, int b) {
    const int p = (n + b - 1) / b;
    for (int jb = 0; jb < p; ++jb) {
        const int j0 = jb * b, mj = std::min(b, n - j0);
        for (int ib = jb; ib < p; ++ib) {  // сначала диагональный блок, затем поддиагональные
            const int i0 = ib * b, mi = std::min(b, n - i0);
            const bool diagonal = ib == jb;
            double* lij = row(l, n, i0) + j0;

            // Ã_ij = A_ij − Σ_{k<j} L_ik·L_jkᵀ — блочное скалярное произведение; Ã_ij пишется на место L_ij.
            // У диагонального блока нужен только нижний треугольник.
            for (int r = 0; r < mi; ++r) {
                const int cols = diagonal ? r + 1 : mj;
                for (int c = 0; c < cols; ++c) row(lij, n, r)[c] = row(a, n, i0 + r)[j0 + c];
            }
            for (int kb = 0; kb < jb; ++kb) {  // блоки k < j всегда полные (b×b)
                const int k0 = kb * b;
                for (int r = 0; r < mi; ++r) {
                    const double* lik = row(l, n, i0 + r) + k0;
                    const int cols = diagonal ? r + 1 : mj;
                    for (int c = 0; c < cols; ++c) row(lij, n, r)[c] -= dot(lik, row(l, n, j0 + c) + k0, b);
                }
            }

            if (diagonal)
                factor_diagonal_block(lij, n, mj);                          // L_jj
            else
                solve_lower_transposed(lij, n, mi, mj, row(l, n, j0) + j0, n);  // L_ij = Ã_ij·L_jj⁻ᵀ
        }
    }
}

double max_error_vs_exact(const double* l, int n) {
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) err = std::max(err, std::abs(row(l, n, i)[j] - exact_factor_entry(i, j)));
    return err;
}

}  // namespace lab2
