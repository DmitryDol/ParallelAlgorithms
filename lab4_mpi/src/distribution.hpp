// Этап 4: циклическое распределение блочных строк между процессами (без MPI — проверяется обычными тестами).
//
// Блочная строка ib принадлежит процессу ib mod P. Процесс хранит только свои блочные строки A и L — подряд,
// каждая строка матрицы целиком (длина строки хранения n): строки блока ib начинаются с локальной строки
// (ib div P)·b. Неполным может быть только последний блок — он и последний у своего владельца.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cholesky.hpp"

namespace lab4 {

inline int owner(int ib, int procs) { return ib % procs; }
inline int local_block(int ib, int procs) { return ib / procs; }

class BlockRows {
public:
    BlockRows(int n, int b, int rank, int procs) : n_(n), b_(b), rank_(rank), procs_(procs) {
        for (int ib = rank; ib < block_count(n, b); ib += procs) {
            blocks_.push_back(ib);
            rows_ += std::min(b, n - ib * b);
        }
        a_.assign(static_cast<std::size_t>(rows_) * n, 0.0);
        l_.assign(a_.size(), 0.0);
    }

    int n() const { return n_; }
    int b() const { return b_; }
    int rank() const { return rank_; }
    int procs() const { return procs_; }
    const std::vector<int>& blocks() const { return blocks_; }  // свои блочные строки по возрастанию
    int local_rows() const { return rows_; }

    // Смещение начала блочной строки ib (своей) в локальных массивах.
    std::size_t offset(int ib) const {
        return static_cast<std::size_t>(local_block(ib, procs_)) * b_ * n_;
    }
    const double* a_row(int ib) const { return a_.data() + offset(ib); }
    double* l_row(int ib) { return l_.data() + offset(ib); }
    const double* l_row(int ib) const { return l_.data() + offset(ib); }
    std::vector<double>& a() { return a_; }
    std::vector<double>& l() { return l_; }
    const std::vector<double>& l() const { return l_; }

    // Свои строки тестовой матрицы — по формуле, без пересылок.
    void fill_test_matrix() {
        for (int ib : blocks_) {
            const int i0 = ib * b_;
            fill_test_rows(a_.data() + offset(ib), n_, i0, std::min(b_, n_ - i0), n_);
        }
    }

    // Свои строки из целой матрицы (для тестов на произвольных матрицах).
    void take_rows(const std::vector<double>& full) {
        for (int ib : blocks_) {
            const std::size_t i0 = static_cast<std::size_t>(ib) * b_ * n_;
            const std::size_t count = static_cast<std::size_t>(std::min(b_, n_ - ib * b_)) * n_;
            std::copy(full.begin() + i0, full.begin() + i0 + count, a_.begin() + offset(ib));
        }
    }

private:
    int n_, b_, rank_, procs_;
    int rows_ = 0;
    std::vector<int> blocks_;
    std::vector<double> a_, l_;
};

}  // namespace lab4
