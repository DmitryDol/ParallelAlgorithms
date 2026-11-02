// Программа замеров этапа 3: время и погрешность блочного и точечного метода Холецкого на P потоках.
// Вариант сборки (ключи компиляции, OpenMP или автопараллелизация) задаётся в CMakeLists.txt и печатается в выводе.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cholesky.hpp"
#include "cli.hpp"

#ifndef LAB3_VARIANT
#define LAB3_VARIANT "unknown"
#define LAB3_FLAGS ""
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

// Число потоков, на котором фактически работает программа.
int thread_count() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#elif defined(LAB3_AUTOPAR_THREADS) && LAB3_AUTOPAR_THREADS > 0
    return LAB3_AUTOPAR_THREADS;  // автопараллелизация GCC: число потоков задано при сборке
#elif defined(LAB3_AUTOPAR_THREADS) && defined(_MSC_VER)
    // MSVC /Qpar: распараллеленные циклы выполняет среда OpenMP, число потоков — из OMP_NUM_THREADS (0 — не задано)
    char* env = nullptr;
    std::size_t length = 0;
    int threads = 0;
    if (_dupenv_s(&env, &length, "OMP_NUM_THREADS") == 0 && env) {
        threads = std::atoi(env);
        std::free(env);
    }
    return threads;
#else
    return 1;
#endif
}

std::string openmp_version() {
#ifdef _OPENMP
    return std::to_string(_OPENMP);  // например, 201511 — OpenMP 4.5, 200203 — OpenMP 2.0
#else
    return "";
#endif
}

struct Measurement {
    double t_min, t_median;  // время одного разложения, с
    int batch;               // сколько разложений подряд входит в один замер
    int repeats;             // сколько замеров сделано
    double error;            // max |L − L_точн|
};

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void factor(lab3::Mode mode, const std::vector<double>& a, std::vector<double>& l, int n, int b) {
    if (mode == lab3::Mode::point)
        lab3::cholesky_point(a.data(), l.data(), n);
    else if (mode == lab3::Mode::phases)
        lab3::cholesky_block_phases(a.data(), l.data(), n, b);
    else
        lab3::cholesky_block(a.data(), l.data(), n, b);
}

// Замеряется только разложение. Первый запуск — прогрев и калибровка: если разложение короче min_time,
// в один замер входит пачка из batch разложений, а время делится на batch. Замеров не меньше repeats; короткие
// повторяются, пока суммарно не займут min_total секунд, — так кратковременная помеха не портит все замеры точки.
Measurement measure(lab3::Mode mode, const std::vector<double>& a, std::vector<double>& l, int n, int b,
                    int repeats, double min_time, double min_total) {
    constexpr std::size_t kMaxRepeats = 100;
    Clock::time_point start = Clock::now();
    factor(mode, a, l, n, b);
    const double first = seconds_since(start);

    int batch = 1;
    std::vector<double> times;
    if (first >= min_time)
        times.push_back(first);
    else
        batch = static_cast<int>(std::ceil(min_time / std::max(first, 1e-9)));
    double total = first;
    while (static_cast<int>(times.size()) < repeats || (total < min_total && times.size() < kMaxRepeats)) {
        start = Clock::now();
        for (int k = 0; k < batch; ++k) factor(mode, a, l, n, b);
        const double elapsed = seconds_since(start);
        total += elapsed;
        times.push_back(elapsed / batch);
    }

    std::sort(times.begin(), times.end());
    const std::size_t m = times.size();
    const double median = m % 2 ? times[m / 2] : 0.5 * (times[m / 2 - 1] + times[m / 2]);
    return {times.front(), median, batch, static_cast<int>(m), lab3::max_error_vs_exact(l.data(), n)};
}

const char* mode_name(lab3::Mode mode) {
    return mode == lab3::Mode::point ? "point" : mode == lab3::Mode::phases ? "phases" : "block";
}

void apply_threads(int threads) {
    if (threads <= 0) return;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads != thread_count())
        std::fprintf(stderr, "Внимание: в сборке %s число потоков не задаётся параметром -t (используется %d)\n",
                     LAB3_VARIANT, thread_count());
#endif
}

int dump(const lab3::Options& o) {
    const int n = o.sizes.front();
    const std::vector<double> a = lab3::make_test_matrix(n);
    std::vector<double> l(a.size(), 0.0);
    factor(o.modes.front(), a, l, n, o.block);

    std::ofstream f(o.dump);
    if (!f) {
        std::fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", o.dump.c_str());
        return 1;
    }
    f << std::setprecision(17);  // 17 значащих цифр — число double восстанавливается из текста точно
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) f << l[static_cast<std::size_t>(i) * n + j] << (j + 1 < n ? ' ' : '\n');
    std::printf("L записана в %s, погрешность %.3e\n", o.dump.c_str(), lab3::max_error_vs_exact(l.data(), n));
    return 0;
}

int run(const lab3::Options& o) {
    const int threads = thread_count();
    if (o.csv)
        std::printf("variant,flags,compiler,openmp,mode,N,b,threads,repeats,batch,t_min,t_median,gflops,err_L\n");
    else
        // Шапка выровнена вручную: printf считает ширину в байтах, а кириллица в UTF-8 занимает 2 байта.
        std::printf("Сборка: %s (%s), компилятор: %s, потоков: %d\n\n"
                    "     N | режим  |    b |   P |    t_min, с |    t_мед, с |  GFLOP/s | погрешность\n",
                    LAB3_VARIANT, LAB3_FLAGS, compiler().c_str(), threads);

    bool sane = true;
    for (int n : o.sizes) {
        const std::vector<double> a = lab3::make_test_matrix(n);
        std::vector<double> l(a.size(), 0.0);  // память под L выделяется и заполняется до замеров
        for (lab3::Mode mode : o.modes) {
            const Measurement r = measure(mode, a, l, n, o.block, o.repeats, o.min_time, o.min_total);
            const double gflops = std::pow(static_cast<double>(n), 3) / 3.0 / r.t_min / 1e9;
            const std::string b = mode == lab3::Mode::point ? "" : std::to_string(o.block);
            if (o.csv)
                std::printf("%s,%s,\"%s\",%s,%s,%d,%s,%d,%d,%d,%.6e,%.6e,%.4f,%.6e\n", LAB3_VARIANT, LAB3_FLAGS,
                            compiler().c_str(), openmp_version().c_str(), mode_name(mode), n, b.c_str(), threads,
                            r.repeats, r.batch, r.t_min, r.t_median, gflops, r.error);
            else
                std::printf("%6d | %-6s | %4s | %3d | %11.4e | %11.4e | %8.3f | %11.3e\n", n, mode_name(mode),
                            b.c_str(), threads, r.t_min, r.t_median, gflops, r.error);
            std::fflush(stdout);
            if (!(r.error <= kSaneError)) sane = false;
        }
    }
    if (!sane) {
        std::fprintf(stderr, "Ошибка: погрешность больше %.0e — в программе ошибка\n", kSaneError);
        return 3;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // русский текст в консоли Windows
#endif
    try {
        const lab3::Options o = lab3::parse_args(std::vector<std::string>(argv + 1, argv + argc));
        if (o.help) {
            std::printf("%s", lab3::usage().c_str());
            return 0;
        }
        apply_threads(o.threads);
        if (o.info) {
            std::printf("variant=%s\nflags=%s\ncompiler=%s\nopenmp=%s\nthreads=%d\n", LAB3_VARIANT, LAB3_FLAGS,
                        compiler().c_str(), openmp_version().c_str(), thread_count());
            return 0;
        }
        return o.dump.empty() ? run(o) : dump(o);
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "Ошибка: %s\n\n%s", e.what(), lab3::usage().c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Ошибка: %s\n", e.what());
        return 1;
    }
}
