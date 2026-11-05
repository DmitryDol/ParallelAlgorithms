"""
Этап 4: прогон всех замеров → results/measurements.csv, results/platform.txt.

Запуск из корня репозитория после сборки (см. README.md этапа):
    uv run python lab4_mpi/bench.py
    uv run python lab4_mpi/bench.py --procs 1 2 4 8 12 --sizes 500 1000 2000 --block 16

Эксперименты (в этом порядке):
  block_sweep — последовательная программа, несколько b при N, не равных степеням двойки; если b не задан
                (--block), для остальных замеров берётся b с наименьшим средним относительным временем;
  seq         — последовательная программа (алгоритм этапа 2): T_посл(N) для абсолютного ускорения;
  mpi         — параллельная программа: стратегии block и phases для каждого числа процессов P
                (mpiexec -n P cholesky_mpi …).
Повторный запуск перезаписывает результаты целиком.
"""

import argparse
import csv
import datetime
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from programs import (BUILD_DIR, RESULTS_DIR, find_mpiexec, find_program, launch_command, launch_env, mpi_family,
                      program_info)

DEFAULT_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]  # без степеней двойки (конфликтные промахи)
DEFAULT_SWEEP_SIZES = [1000, 2000]
DEFAULT_BLOCKS = [8, 12, 16, 24, 32, 48, 64]
PROC_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def default_procs(logical: int) -> list[int]:
    """1, 2, 3, 4, 6, 8, 12, … не больше числа логических процессоров (и само это число) — как числа потоков
    этапа 3, чтобы ускорения можно было сравнивать."""
    return sorted({p for p in PROC_STEPS if p <= logical} | {logical})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Замеры этапа 4 (MPI): время и погрешность на P процессах.")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR, help="каталог сборки CMake (%(default)s)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="каталог результатов (%(default)s)")
    parser.add_argument("-n", "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="ряд N (%(default)s)")
    parser.add_argument("-p", "--procs", type=int, nargs="+",
                        help="числа процессов (по умолчанию 1, 2, 3, 4, 6, 8, … до числа логических процессоров)")
    parser.add_argument("-b", "--block", type=int, help="размер блока (по умолчанию — подбирается)")
    parser.add_argument("--sweep-sizes", type=int, nargs="+", default=DEFAULT_SWEEP_SIZES,
                        help="N для подбора размера блока (%(default)s)")
    parser.add_argument("--blocks", type=int, nargs="+", default=DEFAULT_BLOCKS,
                        help="размеры блока для подбора (%(default)s)")
    parser.add_argument("-r", "--repeats", type=int, default=3, help="наименьшее число замеров на точку (%(default)s)")
    args = parser.parse_args(argv)
    args.procs = sorted(set(args.procs or default_procs(os.cpu_count() or 1)))
    if min(args.sizes + args.sweep_sizes + args.blocks + args.procs + [args.block or 1, args.repeats]) < 1:
        parser.error("все размеры, числа процессов и повторов должны быть натуральными числами")
    return args


# ---------------------------------------------------------------- характеристики компьютера (как на этапе 2)
def _cache_text(size_kb: float, instances: int) -> str:
    def fmt(kb):
        return f"{kb / 1024:g} МБ".replace(".", ",") if kb >= 1024 else f"{kb:g} КБ"
    return fmt(size_kb) if instances <= 1 else f"{fmt(size_kb)} ({instances} × {fmt(size_kb / instances)})"


def _linux_hardware() -> dict:
    out = subprocess.run(["lscpu"], capture_output=True, text=True, env={**os.environ, "LANG": "C", "LC_ALL": "C"}).stdout
    fields = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in out.splitlines() if ":" in line)}
    info = {"Процессор": fields.get("Model name", "")}
    if "Core(s) per socket" in fields and "Socket(s)" in fields:
        info["Физических ядер"] = str(int(fields["Core(s) per socket"]) * int(fields["Socket(s)"]))
    mhz = fields.get("CPU max MHz")
    if not mhz:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("cpu MHz"):
                mhz = line.split(":", 1)[1].strip()
                break
    if mhz:
        info["Частота процессора"] = f"{float(mhz) / 1000:.2f} ГГц".replace(".", ",")
    units = {"KiB": 1, "MiB": 1024, "GiB": 1024 * 1024, "K": 1, "M": 1024}
    for key, name in (("L1d cache", "Кэш L1 данных"), ("L2 cache", "Кэш L2"), ("L3 cache", "Кэш L3")):
        parts = fields.get(key, "").replace("(", " ").split()
        if len(parts) >= 2 and parts[1] in units:
            instances = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 1
            info[name] = _cache_text(float(parts[0]) * units[parts[1]], instances)
    return info


def _windows_hardware() -> dict:
    query = ("$p = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,MaxClockSpeed,"
             "L2CacheSize,L3CacheSize; "
             "$l1 = (Get-CimInstance Win32_CacheMemory | Where-Object { $_.Level -eq 3 -or $_.Purpose -match 'L1' } | "
             "Measure-Object InstalledSize -Sum).Sum; "
             "[pscustomobject]@{p=$p; l1=$l1} | ConvertTo-Json -Depth 3")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout
    data = json.loads(out)
    cpu, cores = data["p"], int(data["p"]["NumberOfCores"])
    info = {"Процессор": cpu["Name"].strip(), "Физических ядер": str(cores),
            "Частота процессора": f"{cpu['MaxClockSpeed'] / 1000:.2f} ГГц".replace(".", ",")}
    if data.get("l1"):
        info["Кэш L1 (данные и команды)"] = _cache_text(float(data["l1"]), cores)
    if cpu.get("L2CacheSize"):
        info["Кэш L2"] = _cache_text(float(cpu["L2CacheSize"]), cores)
    if cpu.get("L3CacheSize"):
        info["Кэш L3"] = _cache_text(float(cpu["L3CacheSize"]), 1)
    return info


def hardware_info() -> dict:
    try:
        info = _linux_hardware() if sys.platform.startswith("linux") else _windows_hardware() \
            if sys.platform == "win32" else {}
    except (OSError, ValueError, KeyError, TypeError):
        info = {}
    info.setdefault("Процессор", platform.processor() or "неизвестно")
    info["Логических процессоров"] = str(os.cpu_count())
    return info


# ---------------------------------------------------------------- замеры
def choose_block(sweep: dict) -> int:
    """sweep[N][b] = время. b с наименьшим средним временем относительно лучшего при каждом N (при равенстве —
    меньший b); учитываются только b, замеренные при всех N."""
    common = sorted(set.intersection(*(set(times) for times in sweep.values())))
    def mean_relative(b):
        return sum(times[b] / min(times.values()) for times in sweep.values()) / len(sweep)
    return min(common, key=lambda b: (mean_relative(b), b))


def run_csv(command: list, env=None) -> list[dict]:
    result = subprocess.run([*map(str, command), "--csv"], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, **(env or {})})
    if result.returncode != 0:
        sys.exit(f"Ошибка: {' '.join(map(str, command))} завершилась с кодом {result.returncode}\n{result.stderr}")
    return list(csv.DictReader(io.StringIO(result.stdout)))


def main(argv=None):
    args = parse_args(argv)
    seq, mpi = find_program("seq", args.build_dir), find_program("mpi", args.build_dir)
    mpiexec = find_mpiexec(args.build_dir)
    if seq is None or mpi is None or mpiexec is None:
        sys.exit(f"Не найдены программы cholesky_seq, cholesky_mpi или mpiexec (каталог {args.build_dir}).\n"
                 "Сначала соберите этап 4: uv run python lab4_mpi/run_all.py (или cmake -S lab4_mpi -B lab4_mpi/build "
                 "&& cmake --build lab4_mpi/build --config Release); нужна библиотека MPI (README.md этапа).")
    mpi_info = program_info(mpi, mpiexec)
    family = mpi_family(mpi_info.get("mpi", ""))
    hardware = hardware_info()
    cores = int(hardware.get("Физических ядер") or os.cpu_count() or 1)
    env = launch_env(family)

    rows, sweep = [], {}
    sweep_sizes = [str(n) for n in sorted(set(args.sweep_sizes))]
    if args.block:
        block, block_note = args.block, "задан"
    else:
        for b in sorted(set(args.blocks)):
            print(f"[block_sweep] seq: b = {b}, N = {' '.join(sweep_sizes)}", file=sys.stderr, flush=True)
            for row in run_csv([seq, "-n", *sweep_sizes, "-b", b, "-m", "block", "-r", args.repeats]):
                rows.append({"experiment": "block_sweep", **row})
                sweep.setdefault(int(row["N"]), {})[b] = float(row["t_min"])
        block = choose_block(sweep)
        block_note = f"подобран по наименьшему времени последовательной программы при N = {', '.join(sweep_sizes)}"
    print(f"b = {block} ({block_note})", file=sys.stderr, flush=True)

    sizes = [str(n) for n in sorted(set(args.sizes))]
    common = ["-n", *sizes, "-b", block, "-r", args.repeats]
    print(f"[seq] N = {' '.join(sizes)}", file=sys.stderr, flush=True)
    rows += [{"experiment": "seq", **row} for row in run_csv([seq, *common, "-m", "block"])]
    for p in args.procs:
        command = launch_command(mpiexec, p, mpi, [*common, "-m", "block", "phases"], family, cores)
        print(f"[mpi] P = {p}" + (" (--oversubscribe)" if "--oversubscribe" in command else ""),
              file=sys.stderr, flush=True)
        rows += [{"experiment": "mpi", **row} for row in run_csv(command, env)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))  # у seq и mpi одинаковые столбцы
    with open(args.out_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    seq_info = program_info(seq)
    info = [
        f"Дата: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"ОС: {platform.platform()}",
        *(f"{key}: {value}" for key, value in hardware.items()),
        f"Реализация MPI: {mpi_info.get('mpi', '—')}",
        f"Запуск: {mpiexec} -n P" + (" (при P больше числа ядер — с --oversubscribe)" if family == "openmpi" else ""),
        f"Размер блока: {block} ({block_note})",
        f"Параметры: N = {' '.join(sizes)}; b = {block}; P = {' '.join(map(str, args.procs))}; повторов = {args.repeats}",
        "",
        "Сборки:",
        *(f"  {name}: flags={i.get('flags', '')}; compiler={i.get('compiler', '')}; mpi={i.get('mpi', '')}"
          for name, i in (("seq", seq_info), ("mpi", mpi_info))),
    ]
    (args.out_dir / "platform.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"Записано {len(rows)} замеров в {args.out_dir / 'measurements.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
