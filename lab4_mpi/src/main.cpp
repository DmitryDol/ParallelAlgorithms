// Программа замеров этапа 4: время и погрешность блочного метода Холецкого.
//   cholesky_mpi — параллельная программа на MPI (запуск: mpiexec -n P cholesky_mpi …), собирается с LAB4_MPI;
//   cholesky_seq — та же программа без MPI: последовательный алгоритм этапа 2 на целой матрице (T_посл).
// Вариант сборки и ключи компиляции задаются в CMakeLists.txt и печатаются в выводе.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef LAB4_MPI
#include "distributed.hpp"
#endif
#include "cholesky.hpp"
#include "cli.hpp"
#include "distribution.hpp"

#ifndef LAB4_VARIANT
#define LAB4_VARIANT "unknown"
#define LAB4_FLAGS ""
#endif

namespace {

// Погрешность на тестовой матрице порядка 1e-15…1e-13; больше — ошибка в программе.
constexpr double kSaneError = 1e-9;

std::string compiler() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    // _MSC_FULL_VER = 195136260 → «MSVC 19.51.36260»
    return "MSVC " + std::to_string(_MSC_FULL_VER / 10000000) + "." + std::to_string(_MSC_FULL_VER / 100000 % 100) +
           "." + std::to_string(_MSC_FULL_VER % 100000);
#else
    return "unknown";
#endif
}

// ---------------------------------------------------------------- MPI или его отсутствие
int rank = 0, procs = 1;

#ifdef LAB4_MPI
struct Environment {  // MPI_Finalize при любом выходе из main
    Environment(int* argc, char*** argv) {
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);
    }
    ~Environment() { MPI_Finalize(); }
};

double now() { return MPI_Wtime(); }
void synchronize() { MPI_Barrier(MPI_COMM_WORLD); }
double slowest(double t) {  // время замера — у самого медленного процесса, одинаковое на всех процессах
    double r = 0.0;
    MPI_Allreduce(&t, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return r;
}

// Первая строка описания библиотеки: «Open MPI v4.1.6, …», «MPICH Version: 4.2.0», «Microsoft MPI 10.1.12498.18».
std::string mpi_version() {
    char text[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int length = 0;
    MPI_Get_library_version(text, &length);
    std::string line = text;  // до нулевого символа: length может его учитывать
    line = line.substr(0, line.find_first_of("\r\n,"));  // название и версия, без подробностей сборки
    std::string s;
    for (char c : line)  // лишние пробелы («MPICH Version:      4.2.0») и кавычки (значение идёт в CSV) — убрать
        if (c != '"' && !(c == ' ' && (s.empty() || s.back() == ' '))) s += c;
    while (!s.empty() && s.back() == ' ') s.pop_back();
    int major = 0, minor = 0;
    MPI_Get_version(&major, &minor);
    return s + " (MPI " + std::to_string(major) + "." + std::to_string(minor) + ")";
}
#else
struct Environment {
    Environment(int*, char***) {}
};

using Clock = std::chrono::steady_clock;
double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
void synchronize() {}
double slowest(double t) { return t; }
std::string mpi_version() { return ""; }
#endif

// ---------------------------------------------------------------- замер
struct Measurement {
    double t_min, t_median;  // время одного разложения, с
    int batch;               // сколько разложений подряд входит в один замер
    int repeats;             // сколько замеров сделано
};

// Замеряется только разложение. Первый запуск — прогрев и калибровка: если разложение короче min_time, в один замер
// входит пачка из batch разложений, а время делится на batch. Замеров не меньше repeats; короткие повторяются, пока
// суммарно не займут min_total секунд. Перед каждым замером процессы ждут друг друга (барьер), время замера — у самого
// медленного; поэтому все процессы принимают одинаковые решения о числе замеров.
Measurement measure(const std::function<void()>& factor, int repeats, double min_time, double min_total) {
    constexpr std::size_t kMaxRepeats = 100;
    synchronize();
    double start = now();
    factor();
    const double first = slowest(now() - start);

    int batch = 1;
    std::vector<double> times;
    if (first >= min_time)
        times.push_back(first);
    else
        batch = static_cast<int>(std::ceil(min_time / std::max(first, 1e-9)));
    double total = first;
    while (static_cast<int>(times.size()) < repeats || (total < min_total && times.size() < kMaxRepeats)) {
        synchronize();
        start = now();
        for (int k = 0; k < batch; ++k) factor();
        const double elapsed = slowest(now() - start);
        total += elapsed;
        times.push_back(elapsed / batch);
    }
    std::sort(times.begin(), times.end());
    const std::size_t m = times.size();
    const double median = m % 2 ? times[m / 2] : 0.5 * (times[m / 2 - 1] + times[m / 2]);
    return {times.front(), median, batch, static_cast<int>(m)};
}

const char* mode_name(lab4::Mode mode) { return mode == lab4::Mode::block ? "block" : "phases"; }

// ---------------------------------------------------------------- задача: целая матрица или свои строки
#ifdef LAB4_MPI
struct Problem {
    lab4::BlockRows rows;
    Problem(int n, int b) : rows(n, b, rank, procs) { rows.fill_test_matrix(); }  // свои строки A — по формуле
    void factor(lab4::Mode mode) { lab4::cholesky_distributed(rows, mode, MPI_COMM_WORLD); }
    double error() const { return lab4::max_error_distributed(rows, MPI_COMM_WORLD); }
    std::vector<double> full_factor() const { return lab4::gather_factor(rows, MPI_COMM_WORLD); }
};
#else
struct Problem {
    int n, b;
    std::vector<double> a, l;
    Problem(int n_, int b_) : n(n_), b(b_), a(lab4::make_test_matrix(n_)), l(a.size(), 0.0) {}
    void factor(lab4::Mode) { lab4::cholesky_block(a.data(), l.data(), n, b); }
    double error() const { return lab4::max_error_vs_exact(l.data(), n, 0, n, n); }
    std::vector<double> full_factor() const { return l; }
};
#endif

int dump(const lab4::Options& o) {
    const int n = o.sizes.front();
    Problem problem(n, o.block);
    problem.factor(o.modes.front());
    const double err = problem.error();
    const std::vector<double> l = problem.full_factor();
    if (rank != 0) return 0;

    std::ofstream f(o.dump);
    if (!f) {
        std::fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", o.dump.c_str());
        return 1;
    }
    f << std::setprecision(17);  // 17 значащих цифр — число double восстанавливается из текста точно
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) f << l[static_cast<std::size_t>(i) * n + j] << (j + 1 < n ? ' ' : '\n');
    std::printf("L записана в %s, погрешность %.3e\n", o.dump.c_str(), err);
    return 0;
}

int run(const lab4::Options& o) {
    if (rank == 0) {
        if (o.csv)
            std::printf("variant,flags,compiler,mpi,mode,N,b,procs,repeats,batch,t_min,t_median,gflops,err_L\n");
        else
            // Шапка выровнена вручную: printf считает ширину в байтах, а кириллица в UTF-8 занимает 2 байта.
            std::printf("Сборка: %s (%s), компилятор: %s%s, процессов: %d\n\n"
                        "     N | режим  |    b |   P |    t_min, с |    t_мед, с |  GFLOP/s | погрешность\n",
                        LAB4_VARIANT, LAB4_FLAGS, compiler().c_str(),
                        mpi_version().empty() ? "" : (", " + mpi_version()).c_str(), procs);
    }
    bool sane = true;
    for (int n : o.sizes) {
        Problem problem(n, o.block);  // память под A и L выделяется и заполняется до замеров
        for (lab4::Mode mode : o.modes) {
            const Measurement r = measure([&] { problem.factor(mode); }, o.repeats, o.min_time, o.min_total);
            const double error = problem.error();
            const double gflops = std::pow(static_cast<double>(n), 3) / 3.0 / r.t_min / 1e9;
            if (rank == 0) {
                if (o.csv)
                    std::printf("%s,%s,\"%s\",\"%s\",%s,%d,%d,%d,%d,%d,%.6e,%.6e,%.4f,%.6e\n", LAB4_VARIANT, LAB4_FLAGS,
                                compiler().c_str(), mpi_version().c_str(), mode_name(mode), n, o.block, procs,
                                r.repeats, r.batch, r.t_min, r.t_median, gflops, error);
                else
                    std::printf("%6d | %-6s | %4d | %3d | %11.4e | %11.4e | %8.3f | %11.3e\n", n, mode_name(mode),
                                o.block, procs, r.t_min, r.t_median, gflops, error);
                std::fflush(stdout);
            }
            if (!(error <= kSaneError)) sane = false;  // погрешность одинакова на всех процессах
        }
    }
    if (!sane) {
        if (rank == 0) std::fprintf(stderr, "Ошибка: погрешность больше %.0e — в программе ошибка\n", kSaneError);
        return 3;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // русский текст в консоли Windows
#endif
    Environment environment(&argc, &argv);
    try {
        lab4::Options o = lab4::parse_args(std::vector<std::string>(argv + 1, argv + argc));
#ifndef LAB4_MPI
        // Стратегии различаются только рассылками; последовательная программа выполняет блочный алгоритм этапа 2.
        if (std::find(o.modes.begin(), o.modes.end(), lab4::Mode::block) == o.modes.end())
            throw std::invalid_argument("последовательная программа выполняет только блочный алгоритм (-m block)");
        o.modes = {lab4::Mode::block};
#endif
        if (o.help) {
            if (rank == 0) std::printf("%s", lab4::usage().c_str());
            return 0;
        }
        if (o.info) {
            if (rank == 0)
                std::printf("variant=%s\nflags=%s\ncompiler=%s\nmpi=%s\nprocs=%d\n", LAB4_VARIANT, LAB4_FLAGS,
                            compiler().c_str(), mpi_version().c_str(), procs);
            return 0;
        }
        return o.dump.empty() ? run(o) : dump(o);
    } catch (const std::invalid_argument& e) {  // одинаково на всех процессах: аргументы у всех одни и те же
        if (rank == 0) std::fprintf(stderr, "Ошибка: %s\n\n%s", e.what(), lab4::usage().c_str());
        return 2;
    } catch (const std::exception& e) {
        if (rank == 0) std::fprintf(stderr, "Ошибка: %s\n", e.what());
        return 1;
    }
}
