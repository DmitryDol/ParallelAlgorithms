// Модульные тесты этапа 4 без MPI (запуск: ctest): ядро, распределение блочных строк, разбор аргументов.
// Сам параллельный алгоритм проверяется программой tests_mpi под mpiexec.
#include <stdexcept>

#include "cholesky.hpp"
#include "cli.hpp"
#include "distribution.hpp"
#include "test_util.hpp"

using test::bitwise_equal;
using test::ctx;

namespace {

std::vector<double> factor(const std::vector<double>& a, int n, int b) {
    std::vector<double> l(a.size(), 0.0);
    lab4::cholesky_block(a.data(), l.data(), n, b);
    return l;
}

void test_matrix_has_exact_factor() {
    // L_точн · L_точнᵀ = A для тестовой матрицы (проверка формул на малых N).
    for (int n : {1, 2, 5, 13}) {
        const std::vector<double> a = lab4::make_test_matrix(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += lab4::exact_factor_entry(i, k) * lab4::exact_factor_entry(j, k);
                CHECK_CTX(std::abs(s - a[i * n + j]) < 1e-12, ctx(n, 0));
            }
    }
}

void test_fill_test_rows_matches_matrix() {
    // Каждый процесс сам заполняет свои строки A — они должны совпасть со строками целой матрицы.
    const int n = 37;
    const std::vector<double> a = lab4::make_test_matrix(n);
    std::vector<double> rows(5 * n, -1.0);
    lab4::fill_test_rows(rows.data(), n, 30, 5, n);
    CHECK(std::equal(rows.begin(), rows.end(), a.begin() + 30 * n));
}

void test_factor_diagonal_block() {
    const int m = 9;
    const std::vector<double> a = test::random_spd(m, 7);
    std::vector<double> l = a;
    lab4::factor_diagonal_block(l.data(), m, m);
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j) l[i * m + j] = 0.0;
    CHECK(test::residual(l, a, m) < 1e-12 * test::max_abs(a));
}

void test_solve_lower_transposed() {
    // X · Lᵀ = B: подставляем известное X, строим B и решаем.
    const int m = 6, rows = 4, ld_x = 10;
    std::vector<double> l(m * m, 0.0), x_true(rows * m), x(rows * ld_x, 0.0);
    test::Random rnd{3};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) l[i * m + j] = (i == j) ? 2.0 + rnd.next() : rnd.next();
    for (double& v : x_true) v = rnd.next();
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < m; ++t) {
            double s = 0.0;
            for (int k = 0; k <= t; ++k) s += x_true[r * m + k] * l[t * m + k];
            x[r * ld_x + t] = s;
        }
    lab4::solve_lower_transposed(x.data(), ld_x, rows, m, l.data(), m);
    double err = 0.0;
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < m; ++t) err = std::max(err, std::abs(x[r * ld_x + t] - x_true[r * m + t]));
    CHECK(err < 1e-12);
}

void test_test_problem_all_shapes() {
    // Граничные случаи: N = 1, 2; N < b; N = b; N кратно b; N не кратно b; простое N; b = 1; b >= N.
    for (int n : {1, 2, 3, 5, 7, 8, 13, 64, 100, 129}) {
        const std::vector<double> a = lab4::make_test_matrix(n);
        for (int b : {1, 2, 3, 4, 8, 16, 200}) {
            const std::vector<double> l = factor(a, n, b);
            CHECK_CTX(lab4::max_error_vs_exact(l.data(), n, 0, n, n) < 1e-12, ctx(n, b));
            CHECK_CTX(test::upper_is_zero(l, n), ctx(n, b));
        }
    }
}

void test_random_spd_residual() {
    for (int n : {1, 2, 7, 33, 64})
        for (int b : {1, 3, 8, 100}) {
            const std::vector<double> a = test::random_spd(n, 100 + n);
            CHECK_CTX(test::residual(factor(a, n, b), a, n) < 1e-11 * test::max_abs(a), ctx(n, b));
        }
}

void test_not_positive_definite_is_rejected() {
    const std::vector<double> a{1.0, 2.0, 2.0, 1.0};  // собственные числа 3 и −1
    for (int b : {1, 2}) {
        bool thrown = false;
        try {
            factor(a, 2, b);
        } catch (const lab4::NotPositiveDefinite&) {
            thrown = true;
        }
        CHECK_CTX(thrown, ctx(2, b));
    }
}

// Так строку j получают остальные процессы: упакованной в буфер [L_j0 … L_j,j−1 | L_jj] (длины строк j0 и mj).
// Блоки, посчитанные по такому буферу, должны побитно совпасть с блоками, посчитанными на месте в целой матрице.
void test_packed_row_gives_bitwise_same_factor() {
    struct Case { int n, b; };
    for (Case c : {Case{37, 8}, Case{64, 16}, Case{10, 16}, Case{29, 1}}) {
        const std::vector<double> a = test::random_spd(c.n, 900 + c.n);
        const std::vector<double> expected = factor(a, c.n, c.b);
        std::vector<double> l(a.size(), 0.0), buffer;
        const int p = lab4::block_count(c.n, c.b), n = c.n;
        for (int jb = 0; jb < p; ++jb) {
            const int j0 = jb * c.b, mj = std::min(c.b, n - j0);
            double* lj = l.data() + static_cast<std::size_t>(j0) * n;
            lab4::modify_block(a.data() + static_cast<std::size_t>(j0) * n, lj, n, lj, n, n, c.b, jb, jb);
            lab4::factor_block(lj, n, n, c.b, jb);
            buffer.assign(static_cast<std::size_t>(mj) * (j0 + mj), 0.0);
            lab4::pack_row(lj, n, mj, j0, buffer.data());
            const double* off = buffer.data();
            const double* ljj = buffer.data() + static_cast<std::size_t>(mj) * j0;
            for (int ib = jb + 1; ib < p; ++ib) {
                const std::size_t i0 = static_cast<std::size_t>(ib) * c.b * n;
                lab4::modify_block(a.data() + i0, l.data() + i0, n, off, j0, n, c.b, ib, jb);
                lab4::solve_block(l.data() + i0, n, ljj, mj, n, c.b, ib, jb);
            }
        }
        CHECK_CTX(bitwise_equal(l, expected), ctx(c.n, c.b));
    }
}

void test_error_measure() {
    const int n = 4;
    std::vector<double> l(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) l[i * n + j] = lab4::exact_factor_entry(i, j);
    CHECK(lab4::max_error_vs_exact(l.data(), n, 0, n, n) == 0.0);
    l[3 * n + 1] += 1e-3;  // строка 3
    CHECK(std::abs(lab4::max_error_vs_exact(l.data(), n, 0, n, n) - 1e-3) < 1e-15);
    CHECK(lab4::max_error_vs_exact(l.data(), n, 0, 3, n) == 0.0);  // строки 0–2 точные
    // строки 2–3 одного процесса: указатель на строку 2, номер первой строки — 2
    CHECK(std::abs(lab4::max_error_vs_exact(l.data() + 2 * n, n, 2, 2, n) - 1e-3) < 1e-15);
    l[0 * n + 3] = 0.5;  // ненулевой элемент над диагональю — тоже ошибка
    CHECK(std::abs(lab4::max_error_vs_exact(l.data(), n, 0, 1, n) - 0.5) < 1e-15);
}

void test_cyclic_distribution() {
    // Блочная строка i — у процесса i mod P; каждая строка матрицы ровно у одного процесса.
    struct Case { int n, b, procs; };
    for (Case c : {Case{129, 16, 2}, Case{129, 16, 4}, Case{40, 16, 7}, Case{1, 4, 3}, Case{100, 1, 3}}) {
        const int p = lab4::block_count(c.n, c.b);
        int total = 0;
        for (int rank = 0; rank < c.procs; ++rank) {
            const lab4::BlockRows rows(c.n, c.b, rank, c.procs);
            total += rows.local_rows();
            int expected_blocks = 0;
            for (int ib = 0; ib < p; ++ib)
                if (lab4::owner(ib, c.procs) == rank) {
                    ++expected_blocks;
                    // строки блока ib лежат в локальном массиве подряд, с шагом n
                    CHECK(rows.offset(ib) == static_cast<std::size_t>(lab4::local_block(ib, c.procs)) * c.b * c.n);
                }
            CHECK(static_cast<int>(rows.blocks().size()) == expected_blocks);
            CHECK(std::all_of(rows.blocks().begin(), rows.blocks().end(),
                              [&](int ib) { return lab4::owner(ib, c.procs) == rank; }));
        }
        CHECK_CTX(total == c.n, ctx(c.n, c.b) + " P = " + std::to_string(c.procs));
    }
    CHECK(lab4::owner(5, 3) == 2 && lab4::local_block(5, 3) == 1);
    const lab4::BlockRows idle(40, 16, 5, 7);  // p = 3 < P: у процесса 5 нет строк
    CHECK(idle.blocks().empty() && idle.local_rows() == 0);
}

void test_block_rows_take_test_matrix_rows() {
    const int n = 37, b = 8, procs = 3;
    const std::vector<double> a = lab4::make_test_matrix(n);
    for (int rank = 0; rank < procs; ++rank) {
        lab4::BlockRows rows(n, b, rank, procs);
        rows.fill_test_matrix();
        for (int ib : rows.blocks()) {
            const int i0 = ib * b, mi = std::min(b, n - i0);
            CHECK(std::equal(rows.a_row(ib), rows.a_row(ib) + mi * n, a.begin() + i0 * n));
        }
    }
}

void test_parse_args() {
    using lab4::Mode;
    const lab4::Options d = lab4::parse_args({});
    CHECK(d.modes == (std::vector<Mode>{Mode::block, Mode::phases}) && d.block == 16 && !d.csv);

    const lab4::Options o = lab4::parse_args({"-n", "300", "100", "100", "-b", "12", "-m", "phases", "-r", "5", "--csv"});
    CHECK(o.sizes == (std::vector<int>{100, 300}) && o.block == 12 && o.repeats == 5 && o.csv);
    CHECK(o.modes == std::vector<Mode>{Mode::phases});
    CHECK(lab4::parse_args({"-m", "all"}).modes == (std::vector<Mode>{Mode::block, Mode::phases}));
    CHECK(lab4::parse_args({"--min-total", "0"}).min_total == 0.0);
    CHECK(lab4::parse_args({"-n", "50", "-m", "block", "--dump", "l.txt"}).dump == "l.txt");

    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"-n", "0"}, {"-n", "-5"}, {"-n", "x"}, {"-n"}, {"-b", "0"}, {"-b"}, {"-m", "point"}, {"-m"},
             {"-t", "4"}, {"--min-total", "-1"}, {"--dump", "l.txt"}, {"-n", "10", "20", "--dump", "l.txt"},
             {"--unknown"}}) {
        bool thrown = false;
        try {
            lab4::parse_args(bad);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        CHECK_CTX(thrown, bad.empty() ? "" : bad.front());
    }
}

}  // namespace

int main() {
    test_matrix_has_exact_factor();
    test_fill_test_rows_matches_matrix();
    test_factor_diagonal_block();
    test_solve_lower_transposed();
    test_test_problem_all_shapes();
    test_random_spd_residual();
    test_not_positive_definite_is_rejected();
    test_packed_row_gives_bitwise_same_factor();
    test_error_measure();
    test_cyclic_distribution();
    test_block_rows_take_test_matrix_rows();
    test_parse_args();
    if (test::failures) {
        std::fprintf(stderr, "Провалено проверок: %d\n", test::failures);
        return 1;
    }
    std::printf("Все тесты пройдены\n");
    return 0;
}
