"""
Этап 3: прогон всех замеров → results/measurements.csv, results/platform.txt, results/autopar_report.txt.

Запуск из корня репозитория после сборки (см. README.md этапа):
    uv run python lab3_openmp/bench.py
    uv run python lab3_openmp/bench.py --threads 1 2 4 8 12 --sizes 500 1000 2000 --block 16

Эксперименты (в этом порядке):
  block_sweep — сборка seq, блочный режим, несколько b при N, не равных степеням двойки; если b не задан
                (--block), для остальных замеров берётся b с наименьшим средним относительным временем;
  seq         — последовательная сборка: T_посл(N) для абсолютного ускорения;
  omp         — вариант А (директивы OpenMP): режимы block и phases для каждого числа потоков P;
  autopar     — вариант Б (автопараллелизация компилятором) для нескольких P.
Повторный запуск перезаписывает результаты целиком.
"""

import argparse
import csv
import datetime
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from programs import BUILD_DIR, HERE, RESULTS_DIR, autopar_programs, find_program, MSBUILD_LOG

DEFAULT_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]  # без степеней двойки (конфликтные промахи)
DEFAULT_SWEEP_SIZES = [1000, 2000]
DEFAULT_BLOCKS = [8, 12, 16, 24, 32, 48, 64]
THREAD_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def default_threads(logical: int) -> list[int]:
    """1, 2, 3, 4, 6, 8, 12, … не больше числа логических процессоров (и само это число)."""
    return sorted({p for p in THREAD_STEPS if p <= logical} | {logical})


def autopar_threads(threads: list[int]) -> list[int]:
    """Автопараллелизация ускорения не даёт (см. отчёт компилятора) — хватит нескольких P: 2, 4 и наибольшего."""
    return sorted({p for p in threads if p in (2, 4)} | {max(threads)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Замеры этапа 3 (OpenMP): время и погрешность на P потоках.")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR, help="каталог сборки CMake (%(default)s)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="каталог результатов (%(default)s)")
    parser.add_argument("-n", "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="ряд N (%(default)s)")
    parser.add_argument("-t", "--threads", type=int, nargs="+",
                        help="числа потоков (по умолчанию 1, 2, 3, 4, 6, 8, … до числа логических процессоров)")
    parser.add_argument("-b", "--block", type=int, help="размер блока (по умолчанию — подбирается)")
    parser.add_argument("--sweep-sizes", type=int, nargs="+", default=DEFAULT_SWEEP_SIZES,
                        help="N для подбора размера блока (%(default)s)")
    parser.add_argument("--blocks", type=int, nargs="+", default=DEFAULT_BLOCKS,
                        help="размеры блока для подбора (%(default)s)")
    parser.add_argument("-r", "--repeats", type=int, default=3, help="наименьшее число замеров на точку (%(default)s)")
    parser.add_argument("--report-only", action="store_true",
                        help="без замеров: только записать results/autopar_report.txt по журналу сборки")
    args = parser.parse_args(argv)
    args.threads = sorted(set(args.threads or default_threads(os.cpu_count() or 1)))
    if min(args.sizes + args.sweep_sizes + args.blocks + args.threads + [args.block or 1, args.repeats]) < 1:
        parser.error("все размеры, числа потоков и повторов должны быть натуральными числами")
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


def run_program(program: Path, args: list[str], env=None) -> list[dict]:
    result = subprocess.run([str(program), *args, "--csv"], capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, **(env or {})})
    if result.returncode != 0:
        sys.exit(f"Ошибка: {program.name} {' '.join(args)} завершилась с кодом {result.returncode}\n{result.stderr}")
    return list(csv.DictReader(io.StringIO(result.stdout)))


GCC_REASONS = {  # причины, по которым GCC не распараллелил цикл (дамп прохода parloops)
    "data dependencies exist across iterations": "между итерациями есть зависимость по данным",
    "cannot analyze data dependencies": "компилятор не смог проанализировать зависимости (обращения к одному массиву "
                                        "через указатели)",
    "scalar dependency between iterations": "скалярная зависимость между итерациями (накапливаемая сумма)",
    "it is not a part of reduction.": "переменная, переходящая между итерациями, не является редукцией",
}

MSVC_REASONS = {  # коды причин C5002 /Qpar (документация MSVC «Vectorizer and parallelizer messages»)
    "500": "у цикла несколько выходов или заголовок не заканчивается увеличением счётчика",
    "501": "счётчик цикла не локальный или верхняя граница меняется в цикле",
    "502": "счётчик цикла меняется не на +1",
    "503": "в цикле есть обработка исключений или switch",
    "504": "тело цикла может выбросить исключение, требующее уничтожения объекта C++",
    "1000": "в теле цикла есть зависимость по данным",
    "1001": "в цикле есть запись в скалярную переменную, которая используется после цикла",
    "1002": "внутренний цикл уже распараллелен",
    "1003": "в теле цикла вызов встроенной функции, которая может читать или писать память",
    "1004": "в теле цикла скалярная редукция",
    "1005": "указана директива no_parallel",
    "1006": "в функции есть директивы OpenMP",
    "1007": "счётчик или границы цикла — не 32-битные целые со знаком",
    "1008": "по оценке компилятора, в цикле слишком мало работы для распараллеливания",
    "1009": "цикл do-while (распараллеливаются только циклы for)",
    "1010": "условие цикла — «не равно» (!=)",
}


def autopar_report(build_dir: Path) -> str:
    """Что распараллелил компилятор в вычислительном ядре (cholesky.cpp) — с текстом строк исходника, — и почему
    не распараллелил остальное (GCC: дамп прохода parloops; MSVC: коды причин C5002)."""
    source = (HERE / "src" / "cholesky.cpp").read_text(encoding="utf-8").splitlines()
    gcc = build_dir / "autopar_report.txt"
    msvc_logs = [log for log in (build_dir / MSBUILD_LOG, build_dir / "build_log.txt") if log.exists()]
    found, reasons = [], {}
    if gcc.exists():
        for line in gcc.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"cholesky\.cpp:(\d+):\d+: optimized: (parallelizing .*)", line)
            if m:
                found.append((int(m.group(1)), m.group(2)))
        for dump in sorted(build_dir.rglob("*cholesky*parloops2")):
            for line in dump.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r"FAILED: (.*)", line)
                if m:
                    reasons[m.group(1).strip()] = reasons.get(m.group(1).strip(), 0) + 1
            break
    elif msvc_logs:
        # «file.cpp(14) : info C5002: …» (cl) или «file.cpp(14,5): info C5002: … [проект.vcxproj]» (MSBuild);
        # русская версия пишет свой текст и кавычки, поэтому код причины — последнее число в сообщении.
        for log in msvc_logs:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r"cholesky\.cpp\((\d+)(?:,\d+)?\)\s*:[^:]*?\b(C500[12])\s*:\s*(.*)", line)
                if m:
                    message = re.sub(r"\s*\[[^\]]*\.vcxproj\]\s*$", "", m.group(3)).strip()
                    found.append((int(m.group(1)), f"{m.group(2)}: {message}"))
        for _, message in sorted(set(found)):  # MSBuild может повторить строку — каждый цикл считается один раз
            code = re.search(r"(\d{3,4})\D*$", message) if message.startswith("C5002") else None
            if code:
                reason = (f"{MSVC_REASONS[code.group(1)]} (код {code.group(1)})" if code.group(1) in MSVC_REASONS
                          else f"код причины {code.group(1)}")
                reasons[reason] = reasons.get(reason, 0) + 1
    if not found and not reasons:
        text = "Отчёт компилятора об автопараллелизации не найден (нужна сборка через run_all.py).\n"
        evidence = list(dict.fromkeys(line.strip()[:300] for log in msvc_logs
                                      for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                                      if "Qpar" in line or "C500" in line))[:10]
        if evidence:  # что есть в журнале: ключи компиляции и строки с кодами C500x — чтобы найти причину
            text += f"\nСтроки журналов сборки ({', '.join(log.name for log in msvc_logs)}) с «Qpar» или «C500»:\n"
            text += "".join(f"  {line}\n" for line in evidence)
        return text
    lines = ["Автопараллелизация: циклы вычислительного ядра (cholesky.cpp), о которых сообщил компилятор", ""]
    for number, message in sorted(set(found)):
        function = next((m.group(1) for line in reversed(source[:number])
                         if (m := re.match(r"\S[^(]*?(\w+)\(.*\)\s*\{\s*$", line))), "?")
        lines.append(f"cholesky.cpp:{number} (функция {function}): {message}")
        lines.append(f"    {source[number - 1].strip() if 0 < number <= len(source) else ''}")
    if reasons:
        lines += ["", "Почему остальные циклы не распараллелены (число случаев):"]
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            text = f"{GCC_REASONS[reason]} ({reason})" if reason in GCC_REASONS else reason
            lines.append(f"  {count} × {text}")
        if msvc_logs and not gcc.exists():
            lines.append("  (расшифровка кодов — в документации MSVC: «Vectorizer and parallelizer messages»)")
    return "\n".join(lines) + "\n"


def write_autopar_report(build_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "autopar_report.txt").write_text(autopar_report(build_dir), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        write_autopar_report(args.build_dir, args.out_dir)
        print(f"Отчёт об автопараллелизации записан в {args.out_dir / 'autopar_report.txt'}", file=sys.stderr)
        return
    seq, omp = find_program("seq", args.build_dir), find_program("omp", args.build_dir)
    if seq is None or omp is None:
        sys.exit(f"Не найдены программы cholesky_seq и cholesky_omp в {args.build_dir}.\n"
                 "Сначала соберите этап 3: uv run python lab3_openmp/run_all.py (или cmake -S lab3_openmp -B "
                 "lab3_openmp/build && cmake --build lab3_openmp/build --config Release)")
    autopar = autopar_programs(args.build_dir)

    rows, sweep = [], {}
    sweep_sizes = [str(n) for n in sorted(set(args.sweep_sizes))]
    if args.block:
        block, block_note = args.block, "задан"
    else:
        for b in sorted(set(args.blocks)):
            print(f"[block_sweep] seq: b = {b}, N = {' '.join(sweep_sizes)}", file=sys.stderr, flush=True)
            for row in run_program(seq, ["-n", *sweep_sizes, "-b", str(b), "-m", "block", "-r", str(args.repeats)]):
                rows.append({"experiment": "block_sweep", **row})
                sweep.setdefault(int(row["N"]), {})[b] = float(row["t_min"])
        block = choose_block(sweep)
        block_note = f"подобран по наименьшему времени последовательной программы при N = {', '.join(sweep_sizes)}"
    print(f"b = {block} ({block_note})", file=sys.stderr, flush=True)

    sizes = [str(n) for n in sorted(set(args.sizes))]
    common = ["-n", *sizes, "-b", str(block), "-r", str(args.repeats)]
    print(f"[seq] N = {' '.join(sizes)}", file=sys.stderr, flush=True)
    rows += [{"experiment": "seq", **row} for row in run_program(seq, [*common, "-m", "block"])]
    for p in args.threads:
        print(f"[omp] P = {p}", file=sys.stderr, flush=True)
        rows += [{"experiment": "omp", **row} for row in run_program(omp, [*common, "-m", "block", "phases",
                                                                          "-t", str(p)])]
    for p in autopar_threads(args.threads):
        program = autopar.get(p) or autopar.get(0)  # GCC: своя сборка на каждое P; MSVC: одна, P — из окружения
        if program is None:
            continue
        print(f"[autopar] P = {p}: {program.name}", file=sys.stderr, flush=True)
        rows += [{"experiment": "autopar", **row}
                 for row in run_program(program, [*common, "-m", "block"], env={"OMP_NUM_THREADS": str(p)})]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_autopar_report(args.build_dir, args.out_dir)

    info = [
        f"Дата: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"ОС: {platform.platform()}",
        *(f"{key}: {value}" for key, value in hardware_info().items()),
        f"Размер блока: {block} ({block_note})",
        f"Параметры: N = {' '.join(sizes)}; b = {block}; P = {' '.join(map(str, args.threads))}; "
        f"автопараллелизация: P = {' '.join(map(str, autopar_threads(args.threads)))}; повторов = {args.repeats}",
        "",
        "Сборки:",
    ]
    autopar_names = {p: f"autopar_p{p}" if p else "autopar (число потоков — из OMP_NUM_THREADS)" for p in autopar}
    for name, program in [("seq", seq), ("omp", omp), *((autopar_names[p], path) for p, path in autopar.items())]:
        details = subprocess.run([str(program), "--info"], capture_output=True, text=True, encoding="utf-8").stdout
        info.append(f"  {name}: " + "; ".join(details.split("\n")[1:]).strip("; "))
    (args.out_dir / "platform.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"Записано {len(rows)} замеров в {args.out_dir / 'measurements.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
