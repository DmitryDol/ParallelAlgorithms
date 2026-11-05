"""
Тесты вспомогательных функций скриптов этапа 4 (programs.py, bench.py, plot_speedup.py).
    uv run pytest
"""

from pathlib import Path

import pytest

from bench import choose_block, default_procs
from plot_speedup import amdahl_alpha, amdahl_speedup, cyclic_model_speedup
from programs import find_mpiexec, launch_command, launch_env, mpi_family


def test_default_procs():
    assert default_procs(4) == [1, 2, 3, 4]
    assert default_procs(24) == [1, 2, 3, 4, 6, 8, 12, 16, 24]
    assert default_procs(1) == [1]


def test_choose_block():
    assert choose_block({1000: {16: 1.0, 32: 1.1}, 2000: {16: 8.0, 32: 8.1}}) == 16
    assert choose_block({1000: {8: 1.0, 48: 1.0}}) == 8


def test_mpi_family():
    assert mpi_family("Open MPI v4.1.6 (MPI 3.1)") == "openmpi"
    assert mpi_family("MPICH Version: 4.2.0 (MPI 4.1)") == "mpich"
    assert mpi_family("Microsoft MPI 10.1.12498.18 (MPI 2.0)") == "msmpi"
    assert mpi_family("") == "other"


def test_launch_command():
    program = Path("build/cholesky_mpi")
    # Open MPI: без --oversubscribe не запустит больше процессов, чем ядер; с ним меняет привязку процессов к ядрам,
    # поэтому ключ — только когда процессов больше, чем ядер
    assert launch_command("mpiexec", 4, program, ["-n", "100"], "openmpi", cores=4) == \
        ["mpiexec", "-n", "4", str(program), "-n", "100"]
    assert launch_command("mpiexec", 8, program, [], "openmpi", cores=4) == \
        ["mpiexec", "-n", "8", "--oversubscribe", str(program)]
    assert launch_command("mpiexec", 24, program, [], "msmpi", cores=12) == ["mpiexec", "-n", "24", str(program)]


def test_launch_env(monkeypatch):
    # Open MPI не запускает процессы от root без разрешения — так бывает только в облачном контейнере
    assert launch_env("openmpi", is_root=True)["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
    assert launch_env("openmpi", is_root=False) == {}
    assert launch_env("msmpi", is_root=True) == {}


def test_find_mpiexec(tmp_path, monkeypatch):
    mpiexec = tmp_path / "mpiexec.exe"
    mpiexec.write_text("")
    (tmp_path / "CMakeCache.txt").write_text(f"MPIEXEC_EXECUTABLE:FILEPATH={mpiexec.as_posix()}\n", encoding="utf-8")
    assert find_mpiexec(tmp_path) == str(mpiexec.as_posix())
    monkeypatch.setenv("PATH", "")
    assert find_mpiexec(tmp_path / "нет") is None


def test_amdahl_fit_recovers_serial_fraction():
    threads = [1, 2, 4, 8, 12]
    assert amdahl_alpha(threads, [amdahl_speedup(0.05, p) for p in threads]) == pytest.approx(0.05)
    assert amdahl_alpha([1], [1.0]) is None


def test_cyclic_model_bounds():
    # Модель распределения строк i mod P без затрат на обмены: S ≤ P, растёт с P, один блок — нечего делить.
    for mode in ("block", "phases"):
        assert cyclic_model_speedup(1000, 12, 1, mode) == pytest.approx(1.0)
        assert cyclic_model_speedup(10, 16, 8, mode) == pytest.approx(1.0)
        speedups = [cyclic_model_speedup(1000, 12, p, mode) for p in (1, 2, 3, 4, 6, 8, 12, 24)]
        assert all(s <= p for s, p in zip(speedups, (1, 2, 3, 4, 6, 8, 12, 24)))
        assert speedups == sorted(speedups)
    # block: на каждом шаге не больше одной параллельной строки, если блочных строк две
    assert cyclic_model_speedup(32, 16, 4, "block") == pytest.approx(1.0)
    # распределение циклическое — то же число строк на процесс, что у schedule(static): та же модель этапа 3
    assert cyclic_model_speedup(3000, 12, 4, "block") == pytest.approx(3.86, abs=0.01)
    assert cyclic_model_speedup(3000, 12, 12, "phases") > cyclic_model_speedup(3000, 12, 12, "block")
