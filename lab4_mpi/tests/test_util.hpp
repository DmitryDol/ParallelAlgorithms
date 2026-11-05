// Общее для тестов этапа 4 (tests.cpp и tests_mpi.cpp): проверки без внешних фреймворков, случайные СПО-матрицы.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace test {

inline int failures = 0;

inline void check(bool ok, const char* what, const char* file, int line, const std::string& context = "") {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "%s:%d: не выполнено: %s %s\n", file, line, what, context.c_str());
    }
}

#define CHECK(cond) test::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_CTX(cond, ctx) test::check((cond), #cond, __FILE__, __LINE__, (ctx))

inline std::string ctx(int n, int b) { return "(N = " + std::to_string(n) + ", b = " + std::to_string(b) + ")"; }

// Детерминированный генератор чисел из [-1, 1), одинаковый на всех платформах и во всех процессах MPI.
struct Random {
    std::uint64_t state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
    }
};

// Случайная СПО-матрица M·Mᵀ + n·I.
inline std::vector<double> random_spd(int n, std::uint64_t seed) {
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

inline double max_abs(const std::vector<double>& v) {
    double r = 0.0;
    for (double x : v) r = std::max(r, std::abs(x));
    return r;
}

// max |(L·Lᵀ − A)_ij| по нижнему треугольнику.
inline double residual(const std::vector<double>& l, const std::vector<double>& a, int n) {
    double r = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k <= j; ++k) s += l[i * n + k] * l[j * n + k];
            r = std::max(r, std::abs(s - a[i * n + j]));
        }
    return r;
}

inline bool upper_is_zero(const std::vector<double>& l, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (l[i * n + j] != 0.0) return false;
    return true;
}

inline bool bitwise_equal(const std::vector<double>& x, const std::vector<double>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
}

}  // namespace test
