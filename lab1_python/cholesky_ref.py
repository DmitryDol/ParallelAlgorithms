"""
Этап 1 (ЛР №1): эталонная реализация блочного метода Холецкого
через блочные скалярные произведения и исследование погрешности.

Запуск из корня репозитория:
    uv run python lab1_python/cholesky_ref.py
    uv run python lab1_python/cholesky_ref.py --sizes 10 100 500 --block-size 8
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_SIZES = [8, 16, 32, 64, 128, 256]
DEFAULT_BLOCK_SIZE = 4
RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
        d = A_block[i, i] - s_diag
        if d <= 0.0:
            raise ValueError("Матрица не является положительно определённой")
        L_block[i, i] = np.sqrt(d)
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


def growth_orders(sizes, errors):
    """
    Порядок роста погрешности между соседними размерностями:
        p = ln(e(N_{k+1}) / e(N_k)) / ln(N_{k+1} / N_k),
    при удвоении N это p = log2(e(2N) / e(N)). Если погрешность нулевая, p = nan.
    """
    orders = []
    for k in range(len(sizes) - 1):
        if errors[k] > 0.0 and errors[k + 1] > 0.0:
            orders.append(float(np.log(errors[k + 1] / errors[k]) / np.log(sizes[k + 1] / sizes[k])))
        else:
            orders.append(float("nan"))
    return orders


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Блочный метод Холецкого через блочные скалярные произведения: "
        "зависимость погрешности от размерности матрицы."
    )
    parser.add_argument(
        "-n", "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help="размерности матрицы N (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "-b", "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help="размер блока b (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=RESULTS_DIR,
        help="каталог для графика и CSV (по умолчанию: lab1_python/results)",
    )
    args = parser.parse_args(argv)
    if args.block_size < 1 or min(args.sizes) < 1:
        parser.error("N и b должны быть натуральными числами")
    return args


def main(argv=None):
    args = parse_args(argv)
    sizes = sorted(set(args.sizes))
    block_size = args.block_size
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
    orders = growth_orders(sizes, errors_L)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "b", "err_L", "residual", "p_to_next"])
        for k, n in enumerate(sizes):
            p_next = f"{orders[k]:.4f}" if k < len(orders) else ""
            writer.writerow([n, block_size, f"{errors_L[k]:.6e}", f"{errors_res[k]:.6e}", p_next])

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

    # Встроенная таблица порядков p = log2(e(2N) / e(N)); левый верхний угол свободен,
    # так как погрешность растёт с ростом N
    if orders:
        table_height = min(0.075 * (len(orders) + 1), 0.75)
        table_data = [
            [f"{sizes[k]} → {sizes[k+1]}", "—" if np.isnan(orders[k]) else f"{orders[k]:.2f}".replace(".", ",")]
            for k in range(len(orders))
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["Переход N", "Порядок p"],
            cellLoc="center",
            bbox=[0.04, 0.96 - table_height, 0.34, table_height],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("gray")
            if row == 0:
                cell.set_facecolor("#d9e8f5")
                cell.get_text().set_weight("bold")

        doubling = all(sizes[k + 1] == 2 * sizes[k] for k in range(len(orders)))
        ax.text(
            0.21, 0.96 - table_height - 0.06,
            r"$p = \log_2(e(2N) / e(N))$" if doubling
            else r"$p = \ln(e(N_{k+1}) / e(N_k)) \,/\, \ln(N_{k+1} / N_k)$",
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            style="italic",
        )

    plt.tight_layout()
    png_path = args.out_dir / "cholesky_error.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"\nГрафик сохранён в файл {png_path}")
    print(f"Таблица погрешностей сохранена в файл {csv_path}")


if __name__ == "__main__":
    main()
