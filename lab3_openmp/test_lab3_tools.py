"""
Тесты вспомогательных функций скриптов этапа 3 (bench.py, plot_speedup.py).
    uv run pytest
"""

import pytest

from bench import autopar_report, autopar_threads, choose_block, default_threads
from plot_speedup import amdahl_alpha, amdahl_speedup, static_schedule_speedup


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
    assert "1 × в теле цикла есть зависимость по данным (код 1000)" in text


def test_autopar_report_from_msbuild_log(tmp_path):
    # MSBuild: позиция «(строка,столбец)», в конце — проект; русская версия: свой текст и кавычки «"1000"»
    (tmp_path / "build_log.txt").write_text(
        r"  C:\lab\src\cholesky.cpp(41,5): info C5001: loop parallelized [C:\lab\build\core_autopar.vcxproj]" + "\n"
        r"  C:\lab\src\cholesky.cpp(14,5): info C5002: loop not parallelized due to reason '1000' "
        r"[C:\lab\build\core_autopar.vcxproj]" + "\n"
        r'  C:\lab\src\cholesky.cpp(99,13): info C5002: цикл не распараллелен по причине "1008" '
        r"[C:\lab\build\core_autopar.vcxproj]" + "\n"
        r"  C:\lab\src\main.cpp(80,5): info C5002: loop not parallelized due to reason '1000'" + "\n",
        encoding="utf-8")
    text = autopar_report(tmp_path)
    assert "cholesky.cpp:41 (функция make_test_matrix): C5001: loop parallelized\n" in text
    assert "vcxproj" not in text and "main.cpp" not in text
    assert "cholesky.cpp:14 (функция dot)" in text and "cholesky.cpp:99 (функция modify_block)" in text
    assert "1 × в теле цикла есть зависимость по данным (код 1000)" in text
    assert "1 × по оценке компилятора, в цикле слишком мало работы для распараллеливания (код 1008)" in text


def test_autopar_report_unknown_msvc_reason(tmp_path):
    (tmp_path / "build_log.txt").write_text(
        r"C:\lab\src\cholesky.cpp(14) : info C5002: loop not parallelized due to reason '1234'" + "\n",
        encoding="utf-8")
    assert "1 × код причины 1234" in autopar_report(tmp_path)


def test_static_schedule_model_bounds():
    for strategy in ("block", "phases"):
        assert static_schedule_speedup(1000, 12, 1, strategy) == pytest.approx(1.0)
        assert static_schedule_speedup(10, 16, 8, strategy) == pytest.approx(1.0)  # один блок — нечего делить
        speedups = [static_schedule_speedup(1000, 12, p, strategy) for p in (1, 2, 3, 4, 6, 8, 12, 24)]
        assert all(s <= p for s, p in zip(speedups, (1, 2, 3, 4, 6, 8, 12, 24)))
        assert speedups == sorted(speedups)  # больше потоков — не медленнее (без накладных расходов)
    # block: на каждом шаге не больше одной параллельной итерации, если блочных строк две
    assert static_schedule_speedup(32, 16, 4, "block") == pytest.approx(1.0)
    # при большом N число блоков на поток велико — ускорение близко к P
    assert static_schedule_speedup(3000, 12, 4, "block") == pytest.approx(3.86, abs=0.01)
    assert static_schedule_speedup(3000, 12, 12, "phases") > static_schedule_speedup(3000, 12, 12, "block")


def test_autopar_report_missing(tmp_path):
    assert "не найден" in autopar_report(tmp_path)
