"""
Этап 3 целиком одной командой: сборка и тесты C++ → сверка с этапами 1–2 → замеры → графики → черновик отчёта.

Запуск из корня репозитория:
    uv run python lab3_openmp/run_all.py
    uv run python lab3_openmp/run_all.py -G Ninja                    # генератор CMake при первой сборке
    uv run python lab3_openmp/run_all.py --threads 1 2 4 8 12 24     # параметры передаются в bench.py
    uv run python lab3_openmp/run_all.py --skip-bench                # без новых замеров: только графики и отчёт

Вычисления и замер времени делает программа на C++ (cholesky_<сборка>); Python только запускает её, собирает CSV
и строит графики и отчёт. Вывод сборки сохраняется в build/build_log.txt: там отчёт MSVC об автопараллелизации.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from programs import BUILD_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def step(title, command, log=None):
    print(f"\n=== {title}: {' '.join(map(str, command))}", flush=True)
    if log is None:
        result = subprocess.run([str(c) for c in command], cwd=ROOT)
    else:
        result = subprocess.run([str(c) for c in command], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        print(result.stdout + result.stderr, end="")
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        sys.exit(f"Шаг «{title}» завершился с ошибкой (код {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Этап 3: сборка, тесты, замеры, графики и отчёт.")
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
        # Полная пересборка, чтобы отчёт компилятора об автопараллелизации попал в журнал сборки.
        step("Сборка", ["cmake", "--build", BUILD_DIR, "--config", "Release", "--clean-first"],
             log=BUILD_DIR / "build_log.txt")
        step("Тесты C++", ["ctest", "--test-dir", BUILD_DIR, "-C", "Release", "--output-on-failure"])
        step("Сверка с этапами 1–2", [python, "-m", "pytest", "-q", HERE / "test_lab3_crosscheck.py"])
    if not args.skip_bench:
        step("Замеры", [python, HERE / "bench.py", *bench_args])
    step("Графики", [python, HERE / "plot_speedup.py"])
    report = HERE / "report" / "make_report.py"
    if report.exists():
        step("Черновик отчёта", [python, report])
    print(f"\nГотово: графики — {HERE / 'results'}" + (f", отчёт — {report.parent}" if report.exists() else ""))


if __name__ == "__main__":
    main()
