// Модульные тесты этапа 2 (запуск: ctest). Без внешних фреймворков, чтобы собиралось и в Windows, и в Linux.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cholesky.hpp"
#include "cli.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what, int line, const std::string& context = "") {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "tests.cpp:%d: не выполнено: %s %s\n", line, what, context.c_str());
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_CTX(cond, ctx) check((cond), #cond, __LINE__, (ctx))

std::string ctx(int n, int b) { return "(N = " + std::to_string(n) + ", b = " + std::to_string(b) + ")"; }

// Детерминированный генератор чисел из [-1, 1), одинаковый на всех платформах.
struct Random {
    std::uint64_t state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
    }
};

// Случайная СПО-матрица M·Mᵀ + n·I.
std::vector<double> random_spd(int n, std::uint64_t seed) {
    Random rnd{seed};
    std::vector<double> m(static_cast<std::size_t>(n) * n);
    for (double& x : m) x = rnd.next();
    std::vector<double> a(m.size());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = (i == j) ? n : 0.0;
            for (int k = 0; k < n; ++k) s += m[i * n + k] * m[j * n + k];
            a[i * n + j] = s;
        }
    return a;
}

double max_abs(const std::vector<double>& v) {
    double r = 0.0;
    for (double x : v) r = std::max(r, std::abs(x));
    return r;
}

// max |(L·Lᵀ − A)_ij| по нижнему треугольнику.
double residual(const std::vector<double>& l, const std::vector<double>& a, int n) {
    double r = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k <= j; ++k) s += l[i * n + k] * l[j * n + k];
            r = std::max(r, std::abs(s - a[i * n + j]));
        }
    return r;
}

bool upper_is_zero(const std::vector<double>& l, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (l[i * n + j] != 0.0) return false;
    return true;
}

std::vector<double> factor(const std::vector<double>& a, int n, int b) {  // b == 0 — точечный алгоритм
    std::vector<double> l(a.size(), 0.0);
    if (b == 0)
        lab2::cholesky_point(a.data(), l.data(), n);
    else
        lab2::cholesky_block(a.data(), l.data(), n, b);
    return l;
}

void test_matrix_has_exact_factor() {
    for (int n : {1, 5, 30}) {
        const std::vector<double> a = lab2::make_test_matrix(n);
        double r = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += lab2::exact_factor_entry(i, k) * lab2::exact_factor_entry(j, k);
                r = std::max(r, std::abs(s - a[i * n + j]));
            }
        CHECK_CTX(r < 1e-12, ctx(n, 0));
    }
    CHECK(lab2::exact_factor_entry(3, 3) == 2.0);
    CHECK(lab2::exact_factor_entry(3, 1) == 1.0 / 3.0);
    CHECK(lab2::exact_factor_entry(1, 3) == 0.0);
}

void test_factor_diagonal_block() {
    // Блок 7×7 внутри массива с длиной строки 10: элементы вне блока не должны меняться.
    const int m = 7, ld = 10;
    const std::vector<double> spd = random_spd(m, 1);
    std::vector<double> buf(ld * ld, 99.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) buf[(i + 1) * ld + (j + 2)] = spd[i * m + j];
    lab2::factor_diagonal_block(&buf[1 * ld + 2], ld, m);

    std::vector<double> l(m * m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) l[i * m + j] = buf[(i + 1) * ld + (j + 2)];
    CHECK(residual(l, spd, m) < 1e-12);
    for (int i = 0; i < m; ++i) CHECK(l[i * m + i] > 0.0);

    bool outside_untouched = true;
    for (int i = 0; i < ld; ++i)
        for (int j = 0; j < ld; ++j) {
            const bool inside_lower = i >= 1 && i < 1 + m && j >= 2 && j - 2 <= i - 1;
            if (!inside_lower && buf[i * ld + j] != 99.0) outside_untouched = false;
        }
    CHECK(outside_untouched);
}

void test_solve_lower_transposed() {
    // X·Lᵀ = B, X — 3×5, хранится с длиной строки 8.
    const int rows = 3, m = 5, ld_x = 8;
    Random rnd{2};
    std::vector<double> l(m * m, 0.0), x_true(rows * m), x(rows * ld_x, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) l[i * m + j] = rnd.next() + (i == j ? 5.0 : 0.0);
    for (double& v : x_true) v = rnd.next();
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < m; ++t) {
            double s = 0.0;
            for (int k = 0; k <= t; ++k) s += x_true[r * m + k] * l[t * m + k];
            x[r * ld_x + t] = s;  // B = X·Lᵀ
        }
    lab2::solve_lower_transposed(x.data(), ld_x, rows, m, l.data(), m);

    double err = 0.0;
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < m; ++t) err = std::max(err, std::abs(x[r * ld_x + t] - x_true[r * m + t]));
    CHECK(err < 1e-12);
}

void test_test_problem_all_shapes() {
    // Граничные случаи: N = 1, 2; N < b; N = b; N кратно b; N не кратно b; простое N; b = 1; b >= N.
    for (int n : {1, 2, 3, 5, 7, 8, 13, 64, 100, 129}) {
        const std::vector<double> a = lab2::make_test_matrix(n);
        for (int b : {0, 1, 2, 3, 4, 8, 16, 200}) {
            const std::vector<double> l = factor(a, n, b);
            CHECK_CTX(lab2::max_error_vs_exact(l.data(), n) < 1e-12, ctx(n, b));
            CHECK_CTX(upper_is_zero(l, n), ctx(n, b));
        }
    }
}

void test_block_matches_point_on_random_spd() {
    for (int n : {1, 2, 7, 33, 64}) {
        const std::vector<double> a = random_spd(n, 100 + n);
        const std::vector<double> lp = factor(a, n, 0);
        CHECK_CTX(residual(lp, a, n) < 1e-11 * max_abs(a), ctx(n, 0));
        for (int b : {1, 3, 8, 100}) {
            const std::vector<double> lb = factor(a, n, b);
            double diff = 0.0;
            for (std::size_t k = 0; k < lb.size(); ++k) diff = std::max(diff, std::abs(lb[k] - lp[k]));
            CHECK_CTX(diff <= 1e-12 * max_abs(lp), ctx(n, b));
        }
    }
}

void test_not_positive_definite_is_rejected() {
    const std::vector<double> a{1.0, 2.0, 2.0, 1.0};  // собственные числа 3 и −1
    for (int b : {0, 1, 2}) {
        bool thrown = false;
        try {
            factor(a, 2, b);
        } catch (const lab2::NotPositiveDefinite&) {
            thrown = true;
        }
        CHECK_CTX(thrown, ctx(2, b));
    }
}

void test_error_measure() {
    const int n = 4;
    std::vector<double> l(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) l[i * n + j] = lab2::exact_factor_entry(i, j);
    CHECK(lab2::max_error_vs_exact(l.data(), n) == 0.0);
    l[2 * n + 1] += 1e-3;
    CHECK(std::abs(lab2::max_error_vs_exact(l.data(), n) - 1e-3) < 1e-15);
    l[0 * n + 3] = 0.5;  // элемент верхнего треугольника тоже учитывается
    CHECK(lab2::max_error_vs_exact(l.data(), n) == 0.5);
}

bool rejected(const std::vector<std::string>& args) {
    try {
        lab2::parse_args(args);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_parse_args() {
    const lab2::Options o = lab2::parse_args({"-n", "512", "100", "100", "-b", "16", "-m", "block", "-r", "5", "--csv"});
    CHECK((o.sizes == std::vector<int>{100, 512}));
    CHECK(o.block == 16);
    CHECK(o.mode == lab2::Mode::block);
    CHECK(o.repeats == 5);
    CHECK(o.csv);

    const lab2::Options d = lab2::parse_args({"--dump", "L.txt", "-n", "37", "-m", "point"});
    CHECK(d.dump == "L.txt");
    CHECK((d.sizes == std::vector<int>{37}));

    CHECK(rejected({"-n", "0"}));
    CHECK(rejected({"-n", "8", "-5"}));
    CHECK(rejected({"-n", "abc"}));
    CHECK(rejected({"-n", "12x"}));
    CHECK(rejected({"-n"}));
    CHECK(rejected({"-b", "0"}));
    CHECK(rejected({"-r", "0"}));
    CHECK(rejected({"-m", "fast"}));
    CHECK(rejected({"--unknown"}));
    CHECK(rejected({"--dump", "L.txt", "-n", "8", "16", "-m", "block"}));  // для выгрузки нужна одна N
    CHECK(rejected({"--dump", "L.txt", "-n", "8"}));                       // и один режим
}

}  // namespace

int main() {
    test_matrix_has_exact_factor();
    test_factor_diagonal_block();
    test_solve_lower_transposed();
    test_test_problem_all_shapes();
    test_block_matches_point_on_random_spd();
    test_not_positive_definite_is_rejected();
    test_error_measure();
    test_parse_args();
    if (failures == 0) std::printf("Все тесты пройдены\n");
    return failures == 0 ? 0 : 1;
}
