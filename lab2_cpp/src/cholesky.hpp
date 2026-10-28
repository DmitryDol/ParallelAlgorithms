// Этап 2: блочный метод Холецкого через блочные скалярные произведения (C++).
//
// Квадратная матрица n×n хранится одним массивом по строкам: элемент (i, j) лежит в a[i * ld + j],
// где ld — длина строки хранения (для целой матрицы ld = n).
#pragma once

#include <stdexcept>
#include <vector>

namespace lab2 {

// Тестовая задача с известным решением — та же, что на этапе 1 (индексы с нуля):
//   L_ii = 2,  L_ij = 1/3 (i > j);   A_ii = 4 + i/9,  A_ij = 2/3 + min(i, j)/9.
double test_matrix_entry(int i, int j);
double exact_factor_entry(int i, int j);
std::vector<double> make_test_matrix(int n);

// Матрица не является положительно определённой (подкоренное выражение <= 0).
struct NotPositiveDefinite : std::domain_error {
    using std::domain_error::domain_error;
};

// Точечный Холецкий диагонального блока m×m на месте (форма ijk через скалярные произведения):
// читает нижний треугольник Ã_jj и записывает на его место L_jj.
void factor_diagonal_block(double* a, int ld, int m);

// Решение X · L_jjᵀ = B по строкам на месте B (rows × m); l — нижнетреугольный блок L_jj (m×m).
void solve_lower_transposed(double* x, int ld_x, int rows, int m, const double* l, int ld_l);

// Разложение A = L·Lᵀ для массивов n×n. Пишется только нижний треугольник l (вместе с диагональю),
// верхний остаётся таким, каким был (вызывающий заполняет l нулями один раз).
void cholesky_point(const double* a, double* l, int n);         // точечный алгоритм, форма jik
void cholesky_block(const double* a, double* l, int n, int b);  // блочный, блоки b×b

// Погрешность max |L_ij − L_точн_ij| по всем элементам матрицы n×n.
double max_error_vs_exact(const double* l, int n);

}  // namespace lab2
