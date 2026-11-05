"""
Сверка этапа 4: фактор L совпадает с эталоном этапа 1 (до 1e-13) в последовательной и MPI-программе в обеих
стратегиях, а MPI-программа при любом числе процессов даёт L, совпадающую побитно с однопроцессной и с
последовательной программой. Нужны собранные программы и MPI:
    cmake -S lab4_mpi -B lab4_mpi/build && cmake --build lab4_mpi/build --config Release
    uv run pytest
Каталог сборки можно указать переменной окружения LAB4_BUILD_DIR.
"""

import functools
import importlib.util
import os
import subprocess

import numpy as np
import pytest

from programs import BUILD_DIR, HERE, find_mpiexec, find_program, launch_command, launch_env, mpi_family, program_info

_spec = importlib.util.spec_from_file_location("lab1_cholesky_ref", HERE.parent / "lab1_python" / "cholesky_ref.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CASES = [(1, 4), (5, 8), (37, 4), (130, 16)]  # N < b; N не кратно b; P больше числа блоков при малых N
PROCS = [1, 2, 3, 4, 7]                        # в том числе P, не делящее p, и P > числа ядер (облачный контейнер)


@functools.lru_cache(maxsize=1)
def mpi_setup():
    mpi, mpiexec = find_program("mpi"), find_mpiexec()
    if mpi is None or mpiexec is None:
        pytest.skip(f"программа cholesky_mpi или mpiexec не найдены (каталог {BUILD_DIR})")
    family = mpi_family(program_info(mpi, mpiexec).get("mpi", ""))
    return mpi, mpiexec, family


def dump(command, path, env=None):
    subprocess.run([*map(str, command), "--dump", str(path)], check=True, capture_output=True,
                   env={**os.environ, **(env or {})})
    return path.read_bytes()


def dump_mpi(procs, n, b, mode, path):
    mpi, mpiexec, family = mpi_setup()
    command = launch_command(mpiexec, procs, mpi, ["-n", n, "-b", b, "-m", mode], family, os.cpu_count() or 1)
    return dump(command, path, launch_env(family))


@pytest.mark.parametrize("mode", ["block", "phases"])
@pytest.mark.parametrize("n, b", CASES)
def test_programs_match_stage1(mode, n, b, tmp_path):
    A, L_exact = ref.generate_test_matrix(n)
    L_ref = ref.block_cholesky_dot(A, block_size=b)
    paths = []
    for procs in (1, 3):
        dump_mpi(procs, n, b, mode, tmp_path / f"mpi{procs}.txt")
        paths.append(tmp_path / f"mpi{procs}.txt")
    seq = find_program("seq")
    if seq is not None and mode == "block":
        dump([seq, "-n", n, "-b", b, "-m", "block"], tmp_path / "seq.txt")
        paths.append(tmp_path / "seq.txt")
    for path in paths:
        L = np.loadtxt(path).reshape(n, n)
        assert np.max(np.abs(L - L_ref)) <= 1e-13, path.name
        assert np.max(np.abs(L - L_exact)) <= 1e-13, path.name


@pytest.mark.parametrize("mode", ["block", "phases"])
@pytest.mark.parametrize("n, b", CASES)
def test_mpi_is_bitwise_independent_of_procs(mode, n, b, tmp_path):
    one = dump_mpi(1, n, b, mode, tmp_path / "p1.txt")
    for procs in PROCS[1:]:
        assert dump_mpi(procs, n, b, mode, tmp_path / f"p{procs}.txt") == one, f"P = {procs}"
    seq = find_program("seq")
    if seq is not None:  # обе программы вызывают одни и те же функции ядра — совпадает и с последовательной
        assert dump([seq, "-n", n, "-b", b, "-m", "block"], tmp_path / "seq.txt") == one
