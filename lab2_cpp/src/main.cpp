// Программа замеров этапа 2: время и погрешность блочного и точечного метода Холецкого.
// Вариант сборки (ключи компиляции) задаётся в CMakeLists.txt и печатается в выводе.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

#include "cholesky.hpp"
#include "cli.hpp"

#ifndef LAB2_VARIANT
#define LAB2_VARIANT "unknown"
#define LAB2_FLAGS ""
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
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
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

void factor(lab2::Mode mode, const std::vector<double>& a, std::vector<double>& l, int n, int b) {
    if (mode == lab2::Mode::point)
        lab2::cholesky_point(a.data(), l.data(), n);
    else
        lab2::cholesky_block(a.data(), l.data(), n, b);
}

// Замеряется только разложение. Первый запуск — прогрев и калибровка: если разложение короче min_time,
// в один замер входит пачка из batch разложений, а время делится на batch. Замеров не меньше repeats; короткие
// повторяются, пока суммарно не займут min_total секунд, — так кратковременная помеха не портит все замеры точки.
Measurement measure(lab2::Mode mode, const std::vector<double>& a, std::vector<double>& l, int n, int b,
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
    return {times.front(), median, batch, static_cast<int>(m), lab2::max_error_vs_exact(l.data(), n)};
}

const char* mode_name(lab2::Mode mode) { return mode == lab2::Mode::point ? "point" : "block"; }

std::vector<lab2::Mode> modes(lab2::Mode m) {
    if (m == lab2::Mode::both) return {lab2::Mode::point, lab2::Mode::block};
    return {m};
}

int dump(const lab2::Options& o) {
    const int n = o.sizes.front();
    const std::vector<double> a = lab2::make_test_matrix(n);
    std::vector<double> l(a.size(), 0.0);
    factor(o.mode, a, l, n, o.block);

    std::ofstream f(o.dump);
    if (!f) {
        std::fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", o.dump.c_str());
        return 1;
    }
    f << std::setprecision(17);  // 17 значащих цифр — число double восстанавливается из текста точно
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) f << l[static_cast<std::size_t>(i) * n + j] << (j + 1 < n ? ' ' : '\n');
    std::printf("L записана в %s, погрешность %.3e\n", o.dump.c_str(), lab2::max_error_vs_exact(l.data(), n));
    return 0;
}

int run(const lab2::Options& o) {
    if (o.csv)
        std::printf("variant,flags,compiler,mode,N,b,repeats,batch,t_min,t_median,gflops,err_L\n");
    else
        // Шапка выровнена вручную: printf считает ширину в байтах, а кириллица в UTF-8 занимает 2 байта.
        std::printf("Сборка: %s (%s), компилятор: %s\n\n"
                    "     N | режим |    b |    t_min, с |    t_мед, с |  GFLOP/s | погрешность\n",
                    LAB2_VARIANT, LAB2_FLAGS, compiler().c_str());

    bool sane = true;
    for (int n : o.sizes) {
        const std::vector<double> a = lab2::make_test_matrix(n);
        std::vector<double> l(a.size(), 0.0);  // память под L выделяется и заполняется до замеров
        for (lab2::Mode mode : modes(o.mode)) {
            const Measurement r = measure(mode, a, l, n, o.block, o.repeats, o.min_time, o.min_total);
            const double gflops = std::pow(static_cast<double>(n), 3) / 3.0 / r.t_min / 1e9;
            const std::string b = mode == lab2::Mode::block ? std::to_string(o.block) : "";
            if (o.csv)
                std::printf("%s,%s,\"%s\",%s,%d,%s,%d,%d,%.6e,%.6e,%.4f,%.6e\n", LAB2_VARIANT, LAB2_FLAGS,
                            compiler().c_str(), mode_name(mode), n, b.c_str(), r.repeats, r.batch, r.t_min,
                            r.t_median, gflops, r.error);
            else
                std::printf("%6d | %-5s | %4s | %11.4e | %11.4e | %8.3f | %11.3e\n", n, mode_name(mode), b.c_str(),
                            r.t_min, r.t_median, gflops, r.error);
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
        const lab2::Options o = lab2::parse_args(std::vector<std::string>(argv + 1, argv + argc));
        if (o.help) {
            std::printf("%s", lab2::usage().c_str());
            return 0;
        }
        if (o.info) {
            std::printf("variant=%s\nflags=%s\ncompiler=%s\n", LAB2_VARIANT, LAB2_FLAGS, compiler().c_str());
            return 0;
        }
        return o.dump.empty() ? run(o) : dump(o);
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "Ошибка: %s\n\n%s", e.what(), lab2::usage().c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Ошибка: %s\n", e.what());
        return 1;
    }
}
