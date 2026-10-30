"""
Этап 2: прогон всех замеров → results/measurements.csv и results/platform.txt.

Запуск из корня репозитория после сборки (см. README.md этапа):
    uv run python lab2_cpp/bench.py
    uv run python lab2_cpp/bench.py --sizes 64 128 256 512 1024 2048 4096 --block 32

Эксперименты (в этом порядке):
  block_sweep  — сборка opt, блочный режим, несколько b при фиксированных N; если b не задан (--block),
                 для основного ряда берётся b с наименьшим средним относительным временем (choose_block);
                 N для подбора — не степени двойки, иначе выбор искажают конфликтные промахи кэша;
  sizes        — каждая сборка (noopt, opt, opt_novec, opt_fastmath), точечный и блочный режимы, ряд N.
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

from programs import BUILD_DIR, RESULTS_DIR, VARIANTS, find_program

DEFAULT_SIZES = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
DEFAULT_SWEEP_SIZES = [1000, 2000]
DEFAULT_BLOCKS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
REQUIRED_VARIANTS = ["noopt", "opt"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Замеры этапа 2 (C++): время и погрешность для всех сборок.")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR, help="каталог сборки CMake (%(default)s)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="каталог результатов (%(default)s)")
    parser.add_argument("-n", "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="ряд N (%(default)s)")
    parser.add_argument("-b", "--block", type=int, help="размер блока для ряда N (по умолчанию — подбирается)")
    parser.add_argument("--sweep-sizes", type=int, nargs="+", default=DEFAULT_SWEEP_SIZES,
                        help="N для подбора размера блока (%(default)s)")
    parser.add_argument("--blocks", type=int, nargs="+", default=DEFAULT_BLOCKS,
                        help="размеры блока для подбора (%(default)s)")
    parser.add_argument("-r", "--repeats", type=int, default=3, help="замеров на точку (%(default)s)")
    args = parser.parse_args(argv)
    if min(args.sizes + args.sweep_sizes + args.blocks + [args.block or 1, args.repeats]) < 1:
        parser.error("все размеры и число повторов должны быть натуральными числами")
    return args


def _cache_text(size_kb: float, instances: int) -> str:
    """Размер кэша: всего и на одно ядро (как в диспетчере задач Windows указывается «всего»)."""
    def fmt(kb):
        return f"{kb / 1024:g} МБ".replace(".", ",") if kb >= 1024 else f"{kb:g} КБ"
    return fmt(size_kb) if instances <= 1 else f"{fmt(size_kb)} ({instances} × {fmt(size_kb / instances)})"


def _linux_hardware() -> dict:
    out = subprocess.run(["lscpu"], capture_output=True, text=True, env={**os.environ, "LANG": "C", "LC_ALL": "C"}).stdout
    fields = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
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
    """Характеристики компьютера для отчёта; чего узнать не удалось — пропускается."""
    try:
        info = _linux_hardware() if sys.platform.startswith("linux") else _windows_hardware() \
            if sys.platform == "win32" else {}
    except (OSError, ValueError, KeyError, TypeError):
        info = {}
    info.setdefault("Процессор", platform.processor() or "неизвестно")
    info["Логических процессоров"] = str(os.cpu_count())
    return info


def choose_block(sweep: dict) -> int:
    """sweep[N][b] = время. Выбирается b с наименьшим средним временем относительно лучшего при каждом N
    (при равенстве — меньший b); учитываются только b, замеренные при всех N."""
    common = sorted(set.intersection(*(set(times) for times in sweep.values())))
    def mean_relative(b):
        return sum(times[b] / min(times.values()) for times in sweep.values()) / len(sweep)
    return min(common, key=lambda b: (mean_relative(b), b))


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

    rows, sweep = [], {}
    sweep_sizes = [str(n) for n in sorted(set(args.sweep_sizes))]
    for b in sorted(set(args.blocks)):
        print(f"[block_sweep] opt: b = {b}, N = {' '.join(sweep_sizes)}", file=sys.stderr, flush=True)
        for row in run_program(programs["opt"], ["-n", *sweep_sizes, "-b", str(b), "-m", "block",
                                                 "-r", str(args.repeats)]):
            rows.append({"experiment": "block_sweep", **row})
            sweep.setdefault(int(row["N"]), {})[b] = float(row["t_min"])
    block = args.block or choose_block(sweep)
    block_note = "задан" if args.block else f"подобран по наименьшему времени при N = {', '.join(sweep_sizes)}"
    print(f"[sizes] b = {block} ({block_note})", file=sys.stderr, flush=True)

    sizes = [str(n) for n in sorted(set(args.sizes))]
    for variant, program in programs.items():
        print(f"[sizes] {variant}: N = {' '.join(sizes)}", file=sys.stderr, flush=True)
        for row in run_program(program, ["-n", *sizes, "-b", str(block), "-m", "both", "-r", str(args.repeats)]):
            rows.append({"experiment": "sizes", **row})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    info = [
        f"Дата: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"ОС: {platform.platform()}",
        *(f"{key}: {value}" for key, value in hardware_info().items()),
        f"Размер блока: {block} ({block_note})",
        f"Параметры: N = {' '.join(sizes)}; b = {block}; подбор b: N = {' '.join(sweep_sizes)}, "
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
