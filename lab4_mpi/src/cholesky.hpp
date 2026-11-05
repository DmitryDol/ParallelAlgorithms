// Этап 4: вычислительное ядро блочного метода Холецкого через блочные скалярные произведения (без MPI).
//
// Матрица хранится по строкам: элемент (i, j) лежит в a[i * ld + j], где ld — длина строки хранения.
// Операции над блоками получают указатель на начало блочной строки (её первая строка, столбец 0) и ld, поэтому
// одинаково работают с целой матрицей (последовательная программа), со строками одного процесса MPI и с принятой
// строкой j. Порядок операций внутри блока — как на этапах 2–3, поэтому результат совпадает с ними побитно.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace lab4 {

// Тестовая задача с известным решением — та же, что на этапе 1 (индексы с нуля):
//   L_ii = 2,  L_ij = 1/3 (i > j);   A_ii = 4 + i/9,  A_ij = 2/3 + min(i, j)/9.
double test_matrix_entry(int i, int j);
double exact_factor_entry(int i, int j);
std::vector<double> make_test_matrix(int n);
// Строки i0 … i0 + rows − 1 тестовой матрицы порядка n (каждый процесс заполняет только свои строки).
void fill_test_rows(double* a, int ld, int i0, int rows, int n);

// Матрица не является положительно определённой (подкоренное выражение <= 0).
struct NotPositiveDefinite : std::domain_error {
    using std::domain_error::domain_error;
};

inline int block_count(int n, int b) { return (n + b - 1) / b; }  // p = ⌈N/b⌉

// Точечный Холецкий диагонального блока m×m на месте: читает нижний треугольник Ã_jj, пишет L_jj.
void factor_diagonal_block(double* a, int ld, int m);
// Решение X · L_jjᵀ = B по строкам на месте B (rows × m); l — нижнетреугольный блок L_jj (m×m).
void solve_lower_transposed(double* x, int ld_x, int rows, int m, const double* l, int ld_l);

// Ã_ij = A_ij − Σ_{k<j} L_ik·L_jkᵀ на месте L_ij (у диагонального блока — только нижний треугольник).
// a_i, l_i — начало блочной строки i в A и L (одна длина строки ld); l_j — начало блочной строки j (ld_j):
// читаются только её столбцы 0 … jb·b − 1. При ib == jb l_j может совпадать с l_i.
void modify_block(const double* a_i, double* l_i, int ld, const double* l_j, int ld_j, int n, int b, int ib, int jb);
// L_jj на месте Ã_jj; l_j — начало блочной строки j.
void factor_block(double* l_j, int ld, int n, int b, int jb);
// L_ij = Ã_ij · L_jj⁻ᵀ на месте; l_i — начало блочной строки i, l_jj — сам блок L_jj (длина строки ld_jj).
void solve_block(double* l_i, int ld, const double* l_jj, int ld_jj, int n, int b, int ib, int jb);

// Строка j для рассылки: [L_j0 … L_j,j−1 | L_jj] — сначала mj×j0 (длина строки j0), затем mj×mj (длина mj).
// Первая часть известна до разложения L_jj, поэтому её можно разослать раньше (стратегия phases).
void pack_row(const double* l_j, int ld, int mj, int j0, double* out);

// Последовательное разложение A = L·Lᵀ целых матриц n×n (программа этапа 2). Пишется только нижний треугольник l,
// верхний остаётся таким, каким был (вызывающий заполняет l нулями один раз).
void cholesky_block(const double* a, double* l, int n, int b);

// Погрешность max |L − L_точн| по строкам i0 … i0 + rows − 1 матрицы порядка n (все столбцы, включая верхний
// треугольник); l — начало строки i0.
double max_error_vs_exact(const double* l, int ld, int i0, int rows, int n);

}  // namespace lab4
