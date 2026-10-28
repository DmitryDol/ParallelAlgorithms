"""
Этап 2: прогон всех замеров → results/measurements.csv и results/platform.txt.

Запуск из корня репозитория после сборки (см. README.md этапа):
    uv run python lab2_cpp/bench.py
    uv run python lab2_cpp/bench.py --sizes 64 128 256 512 1024 2048 4096 --block 32

Эксперименты:
  sizes        — каждая сборка (noopt, opt, opt_novec, opt_fastmath), точечный и блочный режимы, ряд N;
  block_sweep  — сборка opt, блочный режим, несколько b при фиксированных N (выбор размера блока).
Повторный запуск перезаписывает результаты целиком.
"""

import argparse
import csv
import datetime
import io
import os
import platform
import subprocess
import sys
from pathlib import Path

from programs import BUILD_DIR, RESULTS_DIR, VARIANTS, find_program

DEFAULT_SIZES = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
DEFAULT_SWEEP_SIZES = [1024, 2048]
DEFAULT_BLOCKS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
REQUIRED_VARIANTS = ["noopt", "opt"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Замеры этапа 2 (C++): время и погрешность для всех сборок.")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR, help="каталог сборки CMake (%(default)s)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="каталог результатов (%(default)s)")
    parser.add_argument("-n", "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="ряд N (%(default)s)")
    parser.add_argument("-b", "--block", type=int, default=32, help="размер блока для ряда N (%(default)s)")
    parser.add_argument("--sweep-sizes", type=int, nargs="+", default=DEFAULT_SWEEP_SIZES,
                        help="N для подбора размера блока (%(default)s)")
    parser.add_argument("--blocks", type=int, nargs="+", default=DEFAULT_BLOCKS,
                        help="размеры блока для подбора (%(default)s)")
    parser.add_argument("-r", "--repeats", type=int, default=3, help="замеров на точку (%(default)s)")
    args = parser.parse_args(argv)
    if min(args.sizes + args.sweep_sizes + args.blocks + [args.block, args.repeats]) < 1:
        parser.error("все размеры и число повторов должны быть натуральными числами")
    return args


def cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "неизвестно"


def run_program(program: Path, args: list[str]) -> list[dict]:
    result = subprocess.run([str(program), *args, "--csv"], capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        sys.exit(f"Ошибка: {program.name} {' '.join(args)} завершилась с кодом {result.returncode}\n{result.stderr}")
    return list(csv.DictReader(io.StringIO(result.stdout)))


def main(argv=None):
    args = parse_args(argv)
    programs = {v: find_program(v, args.build_dir) for v in VARIANTS}
    missing = [v for v in REQUIRED_VARIANTS if programs[v] is None]
    if missing:
        sys.exit(f"Не найдены программы {', '.join('cholesky_' + v for v in missing)} в {args.build_dir}.\n"
                 "Сначала соберите этап 2: cmake -S lab2_cpp -B lab2_cpp/build && "
                 "cmake --build lab2_cpp/build --config Release")
    programs = {v: p for v, p in programs.items() if p is not None}

    rows = []
    sizes = [str(n) for n in sorted(set(args.sizes))]
    for variant, program in programs.items():
        print(f"[sizes] {variant}: N = {' '.join(sizes)}", file=sys.stderr, flush=True)
        for row in run_program(program, ["-n", *sizes, "-b", str(args.block), "-m", "both", "-r", str(args.repeats)]):
            rows.append({"experiment": "sizes", **row})

    sweep_sizes = [str(n) for n in sorted(set(args.sweep_sizes))]
    for b in sorted(set(args.blocks)):
        print(f"[block_sweep] opt: b = {b}, N = {' '.join(sweep_sizes)}", file=sys.stderr, flush=True)
        for row in run_program(programs["opt"], ["-n", *sweep_sizes, "-b", str(b), "-m", "block",
                                                 "-r", str(args.repeats)]):
            rows.append({"experiment": "block_sweep", **row})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    info = [
        f"Дата: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"ОС: {platform.platform()}",
        f"Процессор: {cpu_name()}",
        f"Логических процессоров: {os.cpu_count()}",
        f"Параметры: N = {' '.join(sizes)}; b = {args.block}; подбор b: N = {' '.join(sweep_sizes)}, "
        f"b = {' '.join(map(str, sorted(set(args.blocks))))}; повторов = {args.repeats}",
        "",
        "Сборки:",
    ]
    for variant, program in programs.items():
        details = subprocess.run([str(program), "--info"], capture_output=True, text=True, encoding="utf-8").stdout
        info.append(f"  {variant}: " + "; ".join(details.split("\n")[1:]).strip("; "))
    (args.out_dir / "platform.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"Записано {len(rows)} замеров в {args.out_dir / 'measurements.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
