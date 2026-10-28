import numpy as np
import matplotlib.pyplot as plt


def generate_test_matrix(n: int):
    """
    Формирует тестовую симметричную положительно определённую матрицу A
    и её точный нижнетреугольный фактор Холецкого L_exact в аналитическом виде:
        L_ii = 2,  L_ij = 1/3 (при i > j)
        A_ii = 4 + i/9,  A_ij = 2/3 + min(i, j)/9 (при i != j)
    """
    L_exact = np.tril(np.full((n, n), 1.0 / 3.0, dtype=np.float64), k=-1)
    np.fill_diagonal(L_exact, 2.0)

    idx = np.arange(n, dtype=np.float64)
    min_ij = np.minimum.outer(idx, idx)
    A = (2.0 / 3.0) + min_ij / 9.0
    np.fill_diagonal(A, 4.0 + idx / 9.0)
    return A, L_exact


def cholesky_point(A_block: np.ndarray) -> np.ndarray:
    """Точечное разложение Холецкого для диагонального блока b x b."""
    b = A_block.shape[0]
    L_block = np.zeros((b, b), dtype=np.float64)
    for i in range(b):
        for j in range(i):
            s = np.dot(L_block[i, :j], L_block[j, :j])
            L_block[i, j] = (A_block[i, j] - s) / L_block[j, j]
        s_diag = np.dot(L_block[i, :i], L_block[i, :i])
        L_block[i, i] = np.sqrt(A_block[i, i] - s_diag)
    return L_block


def solve_triangular_right(B_block: np.ndarray, L_diag: np.ndarray) -> np.ndarray:
    """
    Решает матричное уравнение L_ij * L_jj^T = B_block относительно блока L_ij
    методом прямой подстановки (так как L_jj — нижнетреугольная матрица).
    """
    m, b = B_block.shape
    L_sub = np.zeros((m, b), dtype=np.float64)
    for j in range(b):
        s = L_sub[:, :j] @ L_diag[j, :j] if j > 0 else 0.0
        L_sub[:, j] = (B_block[:, j] - s) / L_diag[j, j]
    return L_sub


def block_cholesky_dot(A: np.ndarray, block_size: int) -> np.ndarray:
    """
    Блочный метод Холецкого через блочные скалярные произведения (left-looking).
    """
    n = A.shape[0]
    p = (n + block_size - 1) // block_size
    L = np.zeros((n, n), dtype=np.float64)

    for j in range(p):
        j0, j1 = j * block_size, min((j + 1) * block_size, n)

        # 1. Блочное скалярное произведение для диагонального блока (j, j)
        A_jj_tilde = A[j0:j1, j0:j1].copy()
        for k in range(j):
            k0, k1 = k * block_size, min((k + 1) * block_size, n)
            L_jk = L[j0:j1, k0:k1]
            A_jj_tilde -= L_jk @ L_jk.T

        # Факторизация диагонального блока
        L_jj = cholesky_point(A_jj_tilde)
        L[j0:j1, j0:j1] = L_jj

        # 2. Вычисление поддиагональных блоков L_ij (i > j) через блочные скалярные произведения
        for i in range(j + 1, p):
            i0, i1 = i * block_size, min((i + 1) * block_size, n)
            A_ij_tilde = A[i0:i1, j0:j1].copy()
            for k in range(j):
                k0, k1 = k * block_size, min((k + 1) * block_size, n)
                L_ik = L[i0:i1, k0:k1]
                L_jk = L[j0:j1, k0:k1]
                A_ij_tilde -= L_ik @ L_jk.T

            L[i0:i1, j0:j1] = solve_triangular_right(A_ij_tilde, L_jj)

    return L


def main():
    sizes = [8, 16, 32, 64, 128, 256]
    block_size = 4
    errors_L = []
    errors_res = []

    print(f"{'N':>5} | {'max |L_числ - L_точн|':>22} | {'max |L*L^T - A|':>18}")
    print("-" * 52)

    for n in sizes:
        A, L_exact = generate_test_matrix(n)
        L_num = block_cholesky_dot(A, block_size=block_size)

        err_L = np.max(np.abs(L_num - L_exact))
        err_res = np.max(np.abs(L_num @ L_num.T - A))
        errors_L.append(err_L)
        errors_res.append(err_res)
        print(f"{n:5d} | {err_L:22.3e} | {err_res:18.3e}")

    # Оценка порядка роста погрешности округления p = log2(e(2N) / e(N))
    orders = [np.log2(errors_L[k + 1] / errors_L[k]) for k in range(len(sizes) - 1)]

    # Построение графика в стиле примера отчёта
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    ax.loglog(sizes, errors_L, marker="o", linewidth=2.5, markersize=9, color="#1f77b4")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, which="both", linestyle="-", alpha=0.35)

    ax.set_title("Блочный метод Холецкого (через блочные скалярные произведения)", fontsize=15, pad=12)
    ax.set_xlabel("Размерность матрицы N", fontsize=14)
    ax.set_ylabel(r"Погрешность  $\max |L_{числ} - L_{точн}|$", fontsize=14)

    # Встроенная таблица порядков p = log2(e(2N) / e(N))
    table_data = [[f"{sizes[k]} → {sizes[k+1]}", f"{orders[k]:.2f}".replace(".", ",")] for k in range(len(orders))]
    table = ax.table(
        cellText=table_data,
        colLabels=["Переход N", "Порядок p"],
        cellLoc="center",
        bbox=[0.62, 0.14, 0.34, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        if row == 0:
            cell.set_facecolor("#d9e8f5")
            cell.get_text().set_weight("bold")

    ax.text(
        0.79, 0.08,
        r"$p = \log_2(e(2N) / e(N))$",
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig("cholesky_error.png", dpi=200)
    print("\nГрафик сохранён в файл cholesky_error.png")


if __name__ == "__main__":
    main()