"""
Сверка этапа 3: фактор L совпадает с эталоном этапа 1 (до 1e-13) во всех сборках и режимах, а программа omp
при любом числе потоков даёт L, совпадающую побитно с однопоточной. Нужна собранная программа:
    cmake -S lab3_openmp -B lab3_openmp/build && cmake --build lab3_openmp/build --config Release
    uv run pytest
Каталог сборки можно указать переменной окружения LAB3_BUILD_DIR.
"""

import importlib.util
import os
import subprocess

import numpy as np
import pytest

from programs import BUILD_DIR, HERE, autopar_programs, find_program

_spec = importlib.util.spec_from_file_location("lab1_cholesky_ref", HERE.parent / "lab1_python" / "cholesky_ref.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CASES = [(1, 4), (5, 8), (37, 4), (130, 16)]  # N < b, N не кратно b, P больше числа блоков при малых N


def dump(program, n, b, mode, path, threads=None):
    args = [str(program), "-n", str(n), "-b", str(b), "-m", mode, "--dump", str(path)]
    if threads:
        args += ["-t", str(threads)]
    subprocess.run(args, check=True, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": str(threads or 1)})
    return path.read_bytes()


def programs():
    found = {"seq": find_program("seq"), "omp": find_program("omp")}
    found.update({f"autopar_p{p}" if p else "autopar": path for p, path in autopar_programs().items()})
    return {name: path for name, path in found.items() if path}


@pytest.mark.parametrize("mode", ["block", "phases", "point"])
@pytest.mark.parametrize("n, b", CASES)
def test_all_programs_match_stage1(mode, n, b, tmp_path):
    found = programs()
    if not found:
        pytest.skip(f"программы этапа 3 не собраны (каталог {BUILD_DIR})")
    A, L_exact = ref.generate_test_matrix(n)
    L_ref = ref.block_cholesky_dot(A, block_size=b)
    for name, program in found.items():
        path = tmp_path / f"{name}.txt"
        dump(program, n, b, mode, path)
        L = np.loadtxt(path).reshape(n, n)
        assert np.max(np.abs(L - L_ref)) <= 1e-13, name
        assert np.max(np.abs(L - L_exact)) <= 1e-13, name


@pytest.mark.parametrize("mode", ["block", "phases", "point"])
@pytest.mark.parametrize("n, b", CASES)
def test_omp_is_bitwise_independent_of_threads(mode, n, b, tmp_path):
    omp = find_program("omp")
    if omp is None:
        pytest.skip(f"программа cholesky_omp не собрана (каталог {BUILD_DIR})")
    one = dump(omp, n, b, mode, tmp_path / "p1.txt", threads=1)
    for p in (2, 3, 4, 7):
        assert dump(omp, n, b, mode, tmp_path / f"p{p}.txt", threads=p) == one, f"P = {p}"
    assert dump(find_program("seq"), n, b, mode, tmp_path / "seq.txt") == one  # совпадает и с последовательной
