"""
Автотесты эталонной реализации (этап 1). Запуск из корня репозитория:
    uv run pytest
"""

import numpy as np
import pytest

from cholesky_ref import (
    block_cholesky_dot,
    cholesky_point,
    generate_test_matrix,
    growth_orders,
    solve_triangular_right,
)


def random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.standard_normal((n, n))
    return m @ m.T + n * np.eye(n)


@pytest.mark.parametrize("n", [8, 30, 256])
def test_analytic_factor_is_exact(n):
    """L_точн * L_точн^T совпадает с A, а численное решение близко к L_точн."""
    A, L_exact = generate_test_matrix(n)
    assert np.max(np.abs(L_exact @ L_exact.T - A)) < 1e-12

    L = block_cholesky_dot(A, block_size=4)
    assert np.max(np.abs(L - L_exact)) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 7, 8, 64, 129])
@pytest.mark.parametrize("b", [1, 3, 4, 16, 200])
def test_matches_lapack_on_random_spd(n, b):
    """Совпадение с LAPACK (numpy.linalg.cholesky), в т.ч. при N, не кратном b, и b > N."""
    rng = np.random.default_rng(1000 * n + b)
    A = random_spd(n, rng)
    L = block_cholesky_dot(A, block_size=b)
    L_ref = np.linalg.cholesky(A)

    assert np.all(np.triu(L, k=1) == 0.0)
    assert np.all(np.diag(L) > 0.0)
    assert np.max(np.abs(L - L_ref)) <= 1e-13 * np.max(np.abs(L_ref))


def test_cholesky_point():
    A = random_spd(6, np.random.default_rng(1))
    L = cholesky_point(A)
    assert np.allclose(L @ L.T, A, rtol=0.0, atol=1e-12)


def test_solve_triangular_right():
    rng = np.random.default_rng(2)
    L_diag = np.tril(rng.standard_normal((5, 5))) + 5.0 * np.eye(5)
    B = rng.standard_normal((7, 5))
    X = solve_triangular_right(B, L_diag)
    assert np.allclose(X @ L_diag.T, B, rtol=0.0, atol=1e-12)


def test_not_positive_definite_is_rejected():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # собственные числа 3 и -1
    with pytest.raises(ValueError):
        block_cholesky_dot(A, block_size=1)


def test_growth_orders():
    assert growth_orders([8, 16, 32], [1.0, 2.0, 8.0]) == pytest.approx([1.0, 2.0])
    assert growth_orders([10, 100], [1.0, 10.0]) == pytest.approx([1.0])
    assert np.isnan(growth_orders([8, 16], [0.0, 1.0])[0])
