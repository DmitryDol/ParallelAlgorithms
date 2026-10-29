"""
Этап 2 целиком одной командой: сборка и тесты C++ → сверка с этапом 1 → замеры → графики → черновик отчёта.

Запуск из корня репозитория:
    uv run python lab2_cpp/run_all.py
    uv run python lab2_cpp/run_all.py -G Ninja                       # выбрать генератор CMake при первой сборке
    uv run python lab2_cpp/run_all.py --sizes 64 128 256 512 1024 2048 4096   # параметры передаются в bench.py
    uv run python lab2_cpp/run_all.py --skip-bench                   # без новых замеров: только графики и отчёт

Вычисления и замер времени делает программа на C++ (cholesky_<сборка>); Python здесь только запускает её с разными
параметрами, собирает CSV и строит графики и отчёт.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from programs import BUILD_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def step(title, command):
    print(f"\n=== {title}: {' '.join(map(str, command))}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Шаг «{title}» завершился с ошибкой (код {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Этап 2: сборка, тесты, замеры, графики и отчёт.")
    parser.add_argument("-G", "--generator", help="генератор CMake для первой настройки (Ninja, \"MinGW Makefiles\")")
    parser.add_argument("--skip-build", action="store_true", help="не собирать и не тестировать C++")
    parser.add_argument("--skip-bench", action="store_true", help="не перемерять: только графики и отчёт")
    args, bench_args = parser.parse_known_args()  # остальные параметры — для bench.py
    python = sys.executable

    if not args.skip_build:
        configure = ["cmake", "-S", HERE, "-B", BUILD_DIR]
        if args.generator and not (BUILD_DIR / "CMakeCache.txt").exists():
            configure += ["-G", args.generator]
        step("Настройка CMake", configure)
        step("Сборка", ["cmake", "--build", BUILD_DIR, "--config", "Release"])
        step("Тесты C++", ["ctest", "--test-dir", BUILD_DIR, "-C", "Release", "--output-on-failure"])
        step("Сверка с этапом 1", [python, "-m", "pytest", "-q", HERE / "test_lab2_crosscheck.py"])
    if not args.skip_bench:
        step("Замеры", [python, HERE / "bench.py", *bench_args])
    step("Графики", [python, HERE / "plot_speedup.py"])
    step("Черновик отчёта", [python, HERE / "report" / "make_report.py"])
    print(f"\nГотово: графики — {HERE / 'results'}, отчёт — {HERE / 'report'}")


if __name__ == "__main__":
    main()
