// Тесты параллельного алгоритма этапа 4 (запуск: ctest, или mpiexec -n P tests_mpi). Каждый процесс проверяет своё,
// число провалов собирается со всех процессов; код возврата ненулевой, если хоть одна проверка не прошла.
#include <mpi.h>

#include "cholesky.hpp"
#include "distributed.hpp"
#include "test_util.hpp"

using test::ctx;

namespace {

int rank = 0, procs = 1;

std::string where(int n, int b, lab4::Mode mode) {
    return ctx(n, b) + " P = " + std::to_string(procs) + (mode == lab4::Mode::block ? " block" : " phases");
}

std::vector<double> sequential(const std::vector<double>& a, int n, int b) {
    std::vector<double> l(a.size(), 0.0);
    lab4::cholesky_block(a.data(), l.data(), n, b);
    return l;
}

// Фактор, собранный со всех процессов, побитно совпадает с последовательным — при любом P, в том числе P > p
// (у части процессов нет блочных строк) и P, не делящем p; повторное разложение тех же строк — тоже.
void test_bitwise_same_as_sequential() {
    struct Case { int n, b; };
    for (Case c : {Case{129, 16}, Case{40, 16}, Case{37, 8}, Case{200, 7}, Case{10, 16}, Case{1, 4}, Case{29, 1}}) {
        const std::vector<double> a = test::random_spd(c.n, 700 + c.n);
        const std::vector<double> expected = sequential(a, c.n, c.b);
        for (lab4::Mode mode : {lab4::Mode::block, lab4::Mode::phases}) {
            lab4::BlockRows rows(c.n, c.b, rank, procs);
            rows.take_rows(a);
            lab4::cholesky_distributed(rows, mode, MPI_COMM_WORLD);
            const std::vector<double> first = lab4::gather_factor(rows, MPI_COMM_WORLD);
            lab4::cholesky_distributed(rows, mode, MPI_COMM_WORLD);  // так повторяются замеры времени
            const std::vector<double> second = lab4::gather_factor(rows, MPI_COMM_WORLD);
            if (rank == 0) {
                CHECK_CTX(test::bitwise_equal(first, expected), where(c.n, c.b, mode));
                CHECK_CTX(test::bitwise_equal(second, expected), where(c.n, c.b, mode) + " повтор");
            }
        }
    }
}

void test_error_on_test_matrix() {
    for (int n : {1, 37, 100})
        for (lab4::Mode mode : {lab4::Mode::block, lab4::Mode::phases}) {
            lab4::BlockRows rows(n, 8, rank, procs);
            rows.fill_test_matrix();
            lab4::cholesky_distributed(rows, mode, MPI_COMM_WORLD);
            const double err = lab4::max_error_distributed(rows, MPI_COMM_WORLD);
            double from_root = err;
            MPI_Bcast(&from_root, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            CHECK_CTX(err < 1e-12, where(n, 8, mode));
            CHECK_CTX(err == from_root, where(n, 8, mode) + " одинаково на всех процессах");
        }
}

// Не положительно определённая матрица: ошибка возникает у владельца блока, а исключение должны бросить все
// процессы — иначе остальные ждали бы рассылки вечно (тест бы завис).
void test_not_positive_definite_on_every_process() {
    struct Case { int n, b; };
    for (Case c : {Case{2, 1}, Case{37, 8}}) {
        std::vector<double> a = c.n == 2 ? std::vector<double>{1.0, 2.0, 2.0, 1.0} : test::random_spd(c.n, 11);
        if (c.n != 2) a.back() = -1.0;  // последний диагональный элемент: ошибка на последнем блоке
        for (lab4::Mode mode : {lab4::Mode::block, lab4::Mode::phases}) {
            lab4::BlockRows rows(c.n, c.b, rank, procs);
            rows.take_rows(a);
            int thrown = 0;
            try {
                lab4::cholesky_distributed(rows, mode, MPI_COMM_WORLD);
            } catch (const lab4::NotPositiveDefinite&) {
                thrown = 1;
            }
            int everywhere = 0;
            MPI_Allreduce(&thrown, &everywhere, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            CHECK_CTX(everywhere == procs, where(c.n, c.b, mode));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    test_bitwise_same_as_sequential();
    test_error_on_test_matrix();
    test_not_positive_definite_on_every_process();
    int total = 0;
    MPI_Allreduce(&test::failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total)
            std::fprintf(stderr, "Провалено проверок: %d (P = %d)\n", total, procs);
        else
            std::printf("Все тесты пройдены (P = %d)\n", procs);
    }
    MPI_Finalize();
    return total ? 1 : 0;
}
