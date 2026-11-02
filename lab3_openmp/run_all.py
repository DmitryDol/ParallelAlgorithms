"""
Этап 3 целиком одной командой: сборка и тесты C++ → сверка с этапами 1–2 → замеры → графики → черновик отчёта.

Запуск из корня репозитория:
    uv run python lab3_openmp/run_all.py
    uv run python lab3_openmp/run_all.py -G Ninja                    # генератор CMake при первой сборке
    uv run python lab3_openmp/run_all.py --threads 1 2 4 8 12 24     # параметры передаются в bench.py
    uv run python lab3_openmp/run_all.py --skip-bench                # без новых замеров: сборка, тесты, отчёт
                                                                     # компилятора, графики (из старого CSV)

Вычисления и замер времени делает программа на C++ (cholesky_<сборка>); Python только запускает её, собирает CSV
и строит графики и отчёт. Вывод сборки сохраняется в build/build_log.txt, у Visual Studio подробный журнал
MSBuild — в build/msbuild_detailed.log: там отчёт MSVC об автопараллелизации.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from programs import BUILD_DIR, MSBUILD_LOG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG_TAIL = 40  # сколько последних строк журнала сборки показать на экране


def step(title, command, log=None):
    print(f"\n=== {title}: {' '.join(map(str, command))}", flush=True)
    if log is None:
        result = subprocess.run([str(c) for c in command], cwd=ROOT)
    else:  # подробный вывод — в журнал, на экран — только конец (там итог сборки и ошибки)
        result = subprocess.run([str(c) for c in command], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        log.write_text(output, encoding="utf-8")
        print("\n".join(output.splitlines()[-LOG_TAIL:]))
        print(f"(полный вывод — в {log})")
    if result.returncode != 0:
        sys.exit(f"Шаг «{title}» завершился с ошибкой (код {result.returncode})")


def cmake_generator(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    text = cache.read_text(encoding="utf-8", errors="replace") if cache.exists() else ""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def build_command(build_dir: Path = BUILD_DIR) -> list:
    """Полная пересборка: отчёт компилятора об автопараллелизации должен попасть в журнал. MSBuild (генератор
    Visual Studio) распознаёт в выводе компилятора только error и warning, а строки /Qpar-report («info C5012»)
    пишет с низкой важностью — их нет ни на экране, ни с --verbose; поэтому — подробный журнал в файл."""
    command = ["cmake", "--build", str(build_dir), "--config", "Release", "--clean-first"]
    if cmake_generator(build_dir).startswith("Visual Studio"):
        command += ["--", f"-flp:Verbosity=detailed;Encoding=UTF-8;LogFile={build_dir / MSBUILD_LOG}"]
    return command


def main():
    parser = argparse.ArgumentParser(description="Этап 3: сборка, тесты, замеры, графики и отчёт.")
    parser.add_argument("-G", "--generator", help="генератор CMake для первой настройки (Ninja, \"MinGW Makefiles\")")
    parser.add_argument("--skip-build", action="store_true", help="не собирать и не тестировать C++")
    parser.add_argument("--skip-bench", action="store_true",
                        help="не перемерять: сборка, тесты, отчёт компилятора, графики")
    args, bench_args = parser.parse_known_args()  # остальные параметры — для bench.py
    python = sys.executable

    if not args.skip_build:
        configure = ["cmake", "-S", HERE, "-B", BUILD_DIR]
        if args.generator and not (BUILD_DIR / "CMakeCache.txt").exists():
            configure += ["-G", args.generator]
        step("Настройка CMake", configure)
        step("Сборка", build_command(BUILD_DIR), log=BUILD_DIR / "build_log.txt")
        step("Отчёт об автопараллелизации", [python, HERE / "bench.py", "--report-only"])
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
