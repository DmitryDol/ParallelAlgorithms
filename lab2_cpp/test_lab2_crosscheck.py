"""
Сверка этапа 2 с эталоном этапа 1: фактор L, найденный программой на C++, совпадает с L из
lab1_python/cholesky_ref.py на тех же N и b. Нужна собранная программа (см. README.md этапа):
    cmake -S lab2_cpp -B lab2_cpp/build && cmake --build lab2_cpp/build --config Release
    uv run pytest
Каталог сборки можно указать переменной окружения LAB2_BUILD_DIR.
"""

import importlib.util
import subprocess

import numpy as np
import pytest

from programs import BUILD_DIR, HERE, VARIANTS, find_program

_spec = importlib.util.spec_from_file_location("lab1_cholesky_ref", HERE.parent / "lab1_python" / "cholesky_ref.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["block", "point"])
@pytest.mark.parametrize("n, b", [(1, 4), (5, 8), (37, 4), (64, 16), (130, 32)])
def test_factor_matches_stage1(variant, mode, n, b, tmp_path):
    program = find_program(variant)
    if program is None:
        pytest.skip(f"программа cholesky_{variant} не собрана (каталог {BUILD_DIR})")

    dump = tmp_path / "L.txt"
    subprocess.run([str(program), "-n", str(n), "-b", str(b), "-m", mode, "--dump", str(dump)],
                   check=True, capture_output=True)
    L_cpp = np.loadtxt(dump).reshape(n, n)

    A, L_exact = ref.generate_test_matrix(n)
    L_ref = ref.block_cholesky_dot(A, block_size=b)
    assert np.max(np.abs(L_cpp - L_ref)) <= 1e-13
    assert np.max(np.abs(L_cpp - L_exact)) <= 1e-13
