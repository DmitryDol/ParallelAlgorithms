"""
Тесты вспомогательных функций скриптов этапа 3 (bench.py, plot_speedup.py).
    uv run pytest
"""

import pytest

from bench import autopar_report, autopar_threads, choose_block, default_threads
from plot_speedup import amdahl_alpha, amdahl_speedup


def test_default_threads():
    assert default_threads(4) == [1, 2, 3, 4]
    assert default_threads(24) == [1, 2, 3, 4, 6, 8, 12, 16, 24]
    assert default_threads(10) == [1, 2, 3, 4, 6, 8, 10]
    assert default_threads(1) == [1]


def test_autopar_threads():
    assert autopar_threads([1, 2, 3, 4]) == [2, 4]
    assert autopar_threads([1, 2, 3, 4, 6, 8, 12, 16, 24]) == [2, 4, 24]
    assert autopar_threads([1]) == [1]


def test_choose_block():
    assert choose_block({1000: {16: 1.0, 32: 1.1}, 2000: {16: 8.0, 32: 8.1}}) == 16
    assert choose_block({1000: {8: 1.0, 48: 1.0}}) == 8


def test_amdahl_fit_recovers_serial_fraction():
    threads = [1, 2, 4, 8, 12]
    speedups = [amdahl_speedup(0.05, p) for p in threads]
    assert amdahl_alpha(threads, speedups) == pytest.approx(0.05)
    assert amdahl_alpha(threads, threads) == pytest.approx(0.0)  # идеальное ускорение — α = 0
    assert amdahl_alpha([1], [1.0]) is None


def test_autopar_report_from_gcc(tmp_path):
    (tmp_path / "autopar_report.txt").write_text(
        "/x/src/cholesky.cpp:40:23: optimized: parallelizing outer loop 1\n"
        "/x/src/cholesky.cpp:38:21: missed: statement clobbers memory: __builtin_GOMP_parallel (...)\n"
        "/x/src/cholesky.cpp:12:5: optimized: loop vectorized using 32 byte vectors\n", encoding="utf-8")
    dump = tmp_path / "CMakeFiles" / "core_autopar_p2.dir" / "src" / "cholesky.cpp.cpp.168t.parloops2"
    dump.parent.mkdir(parents=True)
    dump.write_text("  FAILED: data dependencies exist across iterations\n"
                    "  FAILED: data dependencies exist across iterations\n"
                    "  FAILED: cannot analyze data dependencies\n", encoding="utf-8")
    text = autopar_report(tmp_path)
    assert "cholesky.cpp:40 (функция make_test_matrix): parallelizing outer loop 1" in text
    assert "vectorized" not in text and "clobbers" not in text
    assert "2 × между итерациями есть зависимость по данным (data dependencies exist across iterations)" in text
    assert "1 × " in text and "cannot analyze data dependencies" in text


def test_autopar_report_from_msvc(tmp_path):
    (tmp_path / "build_log.txt").write_text(
        r"C:\lab\src\cholesky.cpp(40) : info C5001: loop parallelized" + "\n"
        r"C:\lab\src\cholesky.cpp(95) : info C5002: loop not parallelized due to reason '1000'" + "\n",
        encoding="utf-8")
    text = autopar_report(tmp_path)
    assert "cholesky.cpp:40 (функция make_test_matrix): C5001: loop parallelized" in text
    assert "C5002: loop not parallelized due to reason '1000'" in text
    assert "1 × код причины 1000" in text


def test_autopar_report_missing(tmp_path):
    assert "не найден" in autopar_report(tmp_path)
