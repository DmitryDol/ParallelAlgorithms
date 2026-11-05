#include "distributed.hpp"

#include <algorithm>
#include <cstddef>

namespace lab4 {

namespace {

// Разложение L_jj с признаком успеха: исключение нельзя «бросить» другим процессам, поэтому признак рассылается
// вместе с L_jj, и после рассылки все процессы решают одинаково.
double try_factor(double* l_j, int n, int b, int jb) {
    try {
        factor_block(l_j, n, n, b, jb);
        return 0.0;
    } catch (const NotPositiveDefinite&) {
        return 1.0;
    }
}

void throw_if_failed(double status) {
    if (status != 0.0) throw NotPositiveDefinite("Матрица не является положительно определённой");
}

void bcast(double* data, std::size_t count, int root, MPI_Comm comm) {
    if (count > 0) MPI_Bcast(data, static_cast<int>(count), MPI_DOUBLE, root, comm);
}

}  // namespace

void cholesky_distributed(BlockRows& m, Mode mode, MPI_Comm comm) {
    const int n = m.n(), b = m.b(), procs = m.procs(), p = block_count(n, b);
    // Строка j в виде для рассылки ([L_j0 … L_j,j−1 | L_jj], pack_row) и за ней — признак ошибки разложения L_jj.
    std::vector<double> buffer(static_cast<std::size_t>(b) * n + 1);
    for (int jb = 0; jb < p; ++jb) {
        const int root = owner(jb, procs), j0 = jb * b, mj = std::min(b, n - j0);
        const std::size_t off = static_cast<std::size_t>(mj) * j0;  // длина внедиагональной части
        const std::size_t diag = static_cast<std::size_t>(mj) * mj;
        const bool mine = m.rank() == root;
        double* lj = mine ? m.l_row(jb) : nullptr;
        const double* l_off = buffer.data();         // L_j0 … L_j,j−1, длина строки j0
        const double* l_jj = buffer.data() + off;    // L_jj, длина строки mj
        double& status = buffer[off + diag];

        // свои блочные строки ниже j
        auto below = std::upper_bound(m.blocks().begin(), m.blocks().end(), jb);
        if (mode == Mode::block) {
            if (mine) {
                modify_block(m.a_row(jb), lj, n, lj, n, n, b, jb, jb);
                status = try_factor(lj, n, b, jb);
                pack_row(lj, n, mj, j0, buffer.data());
            }
            bcast(buffer.data(), off + diag + 1, root, comm);
            throw_if_failed(status);
            for (auto it = below; it != m.blocks().end(); ++it) {
                modify_block(m.a_row(*it), m.l_row(*it), n, l_off, j0, n, b, *it, jb);
                solve_block(m.l_row(*it), n, l_jj, mj, n, b, *it, jb);
            }
        } else {
            if (mine) pack_row(lj, n, mj, j0, buffer.data());  // внедиагональная часть готова с прошлых шагов
            bcast(buffer.data(), off, root, comm);
            if (mine) modify_block(m.a_row(jb), lj, n, lj, n, n, b, jb, jb);
            for (auto it = below; it != m.blocks().end(); ++it)
                modify_block(m.a_row(*it), m.l_row(*it), n, l_off, j0, n, b, *it, jb);
            if (mine) {
                status = try_factor(lj, n, b, jb);
                pack_row(lj, n, mj, j0, buffer.data());
            }
            bcast(buffer.data() + off, diag + 1, root, comm);
            throw_if_failed(status);
            for (auto it = below; it != m.blocks().end(); ++it) solve_block(m.l_row(*it), n, l_jj, mj, n, b, *it, jb);
        }
    }
}

double max_error_distributed(const BlockRows& m, MPI_Comm comm) {
    double local = 0.0;
    for (int ib : m.blocks()) {
        const int i0 = ib * m.b();
        local = std::max(local, max_error_vs_exact(m.l_row(ib), m.n(), i0, std::min(m.b(), m.n() - i0), m.n()));
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
}

std::vector<double> gather_factor(const BlockRows& m, MPI_Comm comm) {
    const int n = m.n(), b = m.b();
    std::vector<double> full(m.rank() == 0 ? static_cast<std::size_t>(n) * n : 0);
    for (int ib = 0; ib < block_count(n, b); ++ib) {
        const int root = owner(ib, m.procs());
        const std::size_t count = static_cast<std::size_t>(std::min(b, n - ib * b)) * n;
        double* target = m.rank() == 0 ? full.data() + static_cast<std::size_t>(ib) * b * n : nullptr;
        if (root == 0) {
            if (m.rank() == 0) std::copy(m.l_row(ib), m.l_row(ib) + count, target);
        } else if (m.rank() == root) {
            MPI_Send(m.l_row(ib), static_cast<int>(count), MPI_DOUBLE, 0, ib, comm);
        } else if (m.rank() == 0) {
            MPI_Recv(target, static_cast<int>(count), MPI_DOUBLE, root, ib, comm, MPI_STATUS_IGNORE);
        }
    }
    return full;
}

}  // namespace lab4
