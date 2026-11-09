"""
Этап 4: черновик отчёта «Отчёт_ПП_4_Абрамкин_Должиков.docx» из результатов замеров.

Запуск из корня репозитория (после bench.py и plot_speedup.py; run_all.py запускает его сам):
    uv run python lab4_mpi/report/make_report.py

Оформление берётся из отчёта этапа 1: титульный лист (меняется только номер работы), стили, колонтитулы,
вид таблиц, рисунков и листингов. Текст — из report_template.md (его можно править), числа, таблицы и графики —
из lab4_mpi/results/ (measurements.csv, platform.txt, *.png), для сравнения с OpenMP — из lab3_openmp/results/,
если этап 3 замерен на том же процессоре. Формулы LaTeX переводятся в формулы Word программой pandoc (пакет
pypandoc-binary). Разделы «Анализ результатов» собираются из чисел и учитывают реализацию MPI и компилятор; после
нового прогона их нужно перечитать. Разметка шаблона, формулы и вёрстка документа — копия генератора этапа 3.
"""

import csv
import math
import re
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pypandoc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_speedup import (OPENMP_RESULTS, Measurements, amdahl_alpha, cyclic_model_speedup,  # noqa: E402
                          draw_scheme, physical_cores, ratio, read_platform)
from programs import RESULTS_DIR, mpi_family  # noqa: E402

TEMPLATE_DOCX = HERE.parent.parent / "lab1_python" / "report" / "Отчёт_ПП_1_Абрамкин_Должиков.docx"
TEMPLATE_TEXT = HERE / "report_template.md"
OUTPUT = HERE / "Отчёт_ПП_4_Абрамкин_Должиков.docx"
LAB_NUMBER = 4

TABLE_WIDTH = 9638           # ширина таблиц (twips), как в отчёте этапа 1
IMAGE_WIDTH_EMU = 5760000    # ширина рисунков — 16 см
BULLET_NUM_ID = "33"         # маркированный список шаблона
DECIMAL_ABSTRACT_NUM = "0"   # нумерованный список шаблона («1.», «2.»)
MACHINE_EPSILON = 2.220446049250313e-16
MODES = ("block", "phases")
AGREEMENT = 0.9              # замер «совпадает с моделью», если не ниже 90 % от неё


# ---------------------------------------------------------------- числа и слова
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def dec(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def dec_tex(x, digits=2):
    """Десятичная запятая внутри формулы LaTeX: 0{,}89 (без лишнего пробела)."""
    return dec(x, digits).replace(",", "{,}")


def sci(x):
    """2,28·10⁻¹⁴"""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    mant = x / 10**e
    if round(mant, 2) >= 10:
        mant, e = mant / 10, e + 1
    return f"{dec(mant)}·10{str(e).translate(SUPERSCRIPT)}"


def ms(t):
    """Время в миллисекундах, три значащие цифры."""
    v = t * 1000
    return dec(v, max(0, 2 - math.floor(math.log10(v))))


def percent(x):
    """0,0448 → «4,5 %»; 0,00726 → «0,73 %»; 0,000036 → «0,0036 %» (две значащие цифры у малых долей)."""
    v = 100 * x
    return f"{dec(v, 1 if v >= 1 else 1 - math.floor(math.log10(v)))} %"


def slashed(fractions):
    """Доли через «/»: [0,045, 0,021] → «4,5 / 2,1 %»."""
    return " / ".join(percent(x)[:-2] for x in fractions) + " %"


def plural(n, one, few, many):
    """21 цикл, 3 цикла, 11 циклов."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def n_set(ns, sizes):
    """Набор размерностей словами: «при всех $N \\geq 200$», «при $N$ = 100 и 300»."""
    if not ns:
        return "ни при одном $N$"
    if ns == sizes:
        return "при всех $N$"
    if len(ns) > 1 and ns == sizes[sizes.index(ns[0]):]:
        return f"при всех $N \\geq {ns[0]}$"
    return f"при $N$ = {and_list(map(str, ns))}"


def trend(first, last):
    if last > first * 1.1:
        return "растёт"
    if last < first / 1.1:
        return "уменьшается"
    return "почти не меняется"


def trend_text(first, n_first, last, n_last):
    """«растёт с 1,82 при $N = 100$ до 7,04 при $N = 3000$» или «почти не меняется: … и …»."""
    word = trend(first, last)
    if word == "почти не меняется":
        return f"{word}: {dec(first)} при $N = {n_first}$ и {dec(last)} при $N = {n_last}$"
    return f"{word} с {dec(first)} при $N = {n_first}$ до {dec(last)} при $N = {n_last}$"


def n_list(ns):
    return ", ".join(map(str, ns))


def and_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


def n_equals(ns):
    """[1000, 2000] → «$N = 1000$ и $N = 2000$»."""
    return and_list(f"$N = {n}$" for n in ns)


def n_range(ns):
    """[100] → «$N = 100$»; [100, 200, 300] → «$N = 100 \\ldots 300$»."""
    return f"$N = {ns[0]}$" if len(ns) == 1 else f"$N = {ns[0]} \\ldots {ns[-1]}$"


def mb(bytes_):
    """36 000 000 байт → «34 МБ»; 450 000 → «0,43 МБ»."""
    v = bytes_ / 2**20
    return f"{dec(v, 0 if v >= 10 else 1 if v >= 1 else 2)} МБ"


def times_slower(x):
    """32,79 → «в 33 раза»; 1,94 → «в 1,9 раза»."""
    return f"в {dec(x, 0 if x >= 10 else 1)} раза"


def table_procs(procs, cores):
    """Числа процессов для таблиц: 2, 4, 8, число ядер, первое P больше числа ядер и наибольшее."""
    beyond = [p for p in procs if cores and p > cores]
    wanted = {2, 4, 8, max(procs)} | ({cores} if cores else set()) | set(beyond[:1])
    return [p for p in procs if p > 1 and p in wanted]


def agreement_limit(ratios: dict):
    """Наибольшее P, до которого (включительно) все отношения «замер / модель» не ниже AGREEMENT; None — если
    уже при наименьшем P ниже."""
    limit = None
    for p in sorted(ratios):
        if min(ratios[p]) < AGREEMENT:
            break
        limit = p
    return limit


def broadcast_numbers(n, b):
    """Сколько чисел рассылается за разложение: на шаге j — строка j, mj × (j0 + mj) чисел (в обеих стратегиях)."""
    total = 0
    for jb in range(-(-n // b)):
        j0 = jb * b
        mj = min(b, n - j0)
        total += mj * (j0 + mj)
    return total


def apply_conditions(text: str, flags: dict) -> str:
    """Строки между «!if флаг» и «!endif» остаются, только если флаг истинен."""
    out, stack = [], []
    for line in text.splitlines():
        if line.startswith("!if "):
            name = line[4:].strip()
            if name not in flags:
                sys.exit(f"В шаблоне неизвестное условие: {name}")
            stack.append(bool(flags[name]))
        elif line.strip() == "!endif":
            if not stack:
                sys.exit("В шаблоне лишний !endif")
            stack.pop()
        elif all(stack):
            out.append(line)
    if stack:
        sys.exit("В шаблоне не закрыт !if")
    return "\n".join(out) + "\n"


def openmp_results(results_dir: Path, processor: str):
    """Замеры этапа 3 программы omp {(режим, N, P): t_min} и погрешности {N: {err}} — если этап 3 замерен на том же
    процессоре; иначе пусто."""
    path = results_dir / "measurements.csv"
    if not path.exists() or read_platform(results_dir / "platform.txt").get("Процессор") != processor:
        return {}, {}, None
    times, errors, block = {}, {}, None
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["experiment"] == "omp":
                times[r["mode"], int(r["N"]), int(r["threads"])] = float(r["t_min"])
                errors.setdefault(int(r["N"]), set()).add(float(r["err_L"]))
                block = int(r["b"])
    return times, errors, block


# ---------------------------------------------------------------- данные
def hardware_rows(info, m):
    rows = [[key, info[key]] for key in ("Процессор", "Частота процессора", "Физических ядер", "Логических процессоров")
            if key in info]
    rows += [[key, value] for key, value in info.items() if key.startswith("Кэш")]
    rows += [["Операционная система", info.get("ОС", "—")],
             ["Компилятор", m.compiler.get("mpi") or m.compiler.get("seq", "—")],
             ["Реализация MPI", m.mpi or "—"],
             ["Язык", "C++ (стандарт C++17)"]]
    return rows


def compute(m: Measurements, info: dict, results_dir: Path, omp_dir: Path) -> tuple[dict, dict, dict]:
    """Подстановки для шаблона, таблицы и условия (!if)."""
    family = mpi_family(m.mpi)
    cores = physical_cores(info)
    sizes, procs, b = m.sizes, m.procs, m.block
    n0, n1 = sizes[0], sizes[-1]
    fixed = m.fixed_sizes()
    big = fixed[-1]
    within = [p for p in procs if cores is None or p <= cores]  # не больше, чем физических ядер
    p_cores, p_max = max(within), max(procs)
    p_tab = table_procs(procs, cores)
    blocks = lambda n: -(-n // b)  # noqa: E731 — число блочных строк

    def s(mode, n, p):
        return m.s_rel(mode, n, p)

    def gflops(n, t):
        return n**3 / 3 / t / 1e9

    model = {(mode, n, p): cyclic_model_speedup(n, b, p, mode) for mode in MODES for n in fixed for p in within}
    ratios = {p: [s(mode, n, p) / model[mode, n, p] for mode in MODES for n in fixed] for p in within if p > 1}
    p_agree = agreement_limit(ratios)
    fit = [p for p in within if p_agree is None or p <= p_agree]
    alpha = {(mode, n): amdahl_alpha(fit, [s(mode, n, p) for p in fit]) for mode in MODES for n in fixed}
    alpha_model = {(mode, n): amdahl_alpha(fit, [model[mode, n, p] for p in fit]) for mode in MODES for n in fixed}
    alpha_theory = {("block", n): 1.5 * b / n for n in fixed} | {("phases", n): (b / n) ** 2 for n in fixed}
    wide = fit != within
    alpha_cores = {(mode, n): amdahl_alpha(within, [s(mode, n, p) for p in within]) for mode in MODES for n in fixed}

    omp_times, omp_errors, omp_block = openmp_results(omp_dir, info.get("Процессор", ""))
    omp_common = [p for p in procs if all((mode, big, p) in omp_times and (mode, big, 1) in omp_times
                                          for mode in MODES)]
    has_omp = len(omp_common) > 1 and (results_dir / "mpi_vs_omp.png").exists()

    def s_omp(mode, n, p):
        return ratio(omp_times.get((mode, n, 1)), omp_times.get((mode, n, p)))

    v = {
        "compiler": m.compiler.get("mpi") or m.compiler.get("seq", "—"),
        "repeats": str(m.repeats),
        "sizes": n_list(sizes),
        "procs": n_list(procs),
        "p_cores": str(p_cores),
        "fixed_text": n_equals(fixed),
        "big": str(big),
    }
    share4, share12 = 1 - (3 / 4) ** 3, 1 - (1 - 1 / p_cores) ** 3
    v["block_share"] = (f"при $P = 4$ это {percent(share4)} и ускорение не больше {dec(1 / share4, 1)}"
                        + (f", при $P = {p_cores}$ — {percent(share12)} и не больше {dec(1 / share12, 1)}"
                           if p_cores > 4 else ""))

    # --- реализация MPI
    mpi_calls = ("Используются только `MPI_Bcast`, `MPI_Allreduce`, `MPI_Barrier`, `MPI_Wtime` и (для проверок) "
                 "`MPI_Send` и `MPI_Recv` — функции стандарта MPI-1.")
    if family == "msmpi":
        v["mpi_explanation"] = (f"Реализация MPI — {m.mpi}: Microsoft MPI для Windows; программа компонуется "
                                "с библиотекой `msmpi.lib` из MS-MPI SDK и запускается командой `mpiexec -n P`. "
                                + mpi_calls)
    elif family == "openmpi":
        v["mpi_explanation"] = (f"Реализация MPI — {m.mpi}; запуск — `mpiexec -n P`, при $P$ больше числа ядер — "
                                "с ключом `--oversubscribe` (без него Open MPI не запускает больше процессов, чем "
                                "ядер). " + mpi_calls)
    else:
        v["mpi_explanation"] = f"Реализация MPI — {m.mpi or 'не указана'}; запуск — `mpiexec -n P`. " + mpi_calls

    # --- выбор размера блока
    sweep_n = sorted(m.sweep)
    v["block_choice"] = (f"Размер блока $b = {b}$ подобран по наименьшему времени последовательной программы при "
                         f"{n_equals(sweep_n)} (таблица <<tab:sweep>>) и используется во всех замерах"
                         if sweep_n else f"Размер блока задан вручную: $b = {b}$ (подбор не выполнялся, таблица "
                                         "<<tab:sweep>>)")

    # --- корректность
    err_by_n = {}
    for (experiment, mode, n, p), e in m.error.items():
        err_by_n.setdefault(n, set()).add(e)
    identical = all(len(errs) == 1 for errs in err_by_n.values())
    e0, e1 = min(err_by_n[n0]), max(err_by_n[n1])
    e_max = max(max(errs) for errs in err_by_n.values())
    same_as_omp = (omp_block == b and all(omp_errors.get(n) == err_by_n[n] for n in sizes)) if omp_errors else False
    v["errors_note"] = (
        ("При каждом $N$ погрешность одна и та же в обеих программах, при всех числах процессов и в обеих стратегиях — "
         "до последнего разряда: процессы считают независимые блоки, а порядок операций внутри блока не меняется."
         if identical else
         "Погрешность в разных программах и при разном числе процессов различается только в последних разрядах "
         "(в таблице — наименьшее и наибольшее значения).")
        + (f" Она совпадает и с погрешностью программы OpenMP лабораторной работы № 3 при том же $b = {b}$."
           if same_as_omp else ""))
    growth = "растёт с увеличением $N$" if e1 > e0 else "почти не зависит от $N$"
    v["analysis_correctness"] = (
        f"Во всём диапазоне $N = {n0} \\ldots {n1}$ погрешность составляет от {sci(e0)} до {sci(e1)} и, как на "
        f"этапах 1–3, {growth} из-за накопления ошибок округления в скалярных произведениях; наибольшая — не более "
        f"{round(e_max / MACHINE_EPSILON)} машинных эпсилон. "
        + ("Распределение данных и обмены не изменили ни одного разряда результата: погрешность одна и та же при "
           "любом числе процессов и в обеих стратегиях, а тесты подтверждают побитное совпадение фактора $L$ "
           "с последовательной программой. " if identical else "")
        + "Ускорение получено без потери точности.")

    # --- объём обменов
    comm_rows = []
    for n in sizes:
        numbers = broadcast_numbers(n, b)
        comm_rows.append([str(n), str(blocks(n)), str(blocks(n)), str(2 * blocks(n)), f"{numbers:,}".replace(",", " "),
                          mb(8 * numbers), dec(n**3 / 3 / numbers, 0)])

    # --- зависимость от N
    s_ref = {n: s("block", n, p_cores) for n in sizes}
    best_p = {n: max(procs, key=lambda p: s("block", n, p)) for n in sizes}
    groups = []
    for n in sizes:
        if groups and groups[-1][0] == best_p[n]:
            groups[-1][1].append(n)
        else:
            groups.append((best_p[n], [n]))
    n_best = max(sizes, key=lambda n: s("block", n, best_p[n]))
    p0 = blocks(n0)
    more = [p for p in procs if p > p0]
    text = (f"При $P = {p_cores}$ относительное ускорение стратегии `block` "
            f"{trend_text(s_ref[n0], n0, s_ref[n1], n1)} (таблица <<tab:speedup_block>>, рис. <<fig:speedup_n.png>>). "
            + ("Наибольшее ускорение достигается " + ", ".join(f"при {n_range(ns)} — при $P = {p}$" for p, ns in groups)
               + ". " if len(groups) > 1 else f"Наибольшее ускорение при всех $N$ достигается при $P = {groups[0][0]}$. ")
            + f"Самое большое значение — {dec(s('block', n_best, best_p[n_best]))} при $N = {n_best}$, "
              f"$P = {best_p[n_best]}$. С ростом $N$ вычисления растут как $N^3$, а пересылки — как $N^2$ "
              "(п. «Объём вычислений и пересылок»), поэтому доля обменов и ожидания уменьшается. Кроме того, при малых "
              f"$N$ блоков мало: при $N = {n0}$ блочных строк всего $p = {p0}$")
    text += (f", меньше, чем процессов при $P \\geq {more[0]}$: у части процессов нет строк, но в каждой рассылке "
             "участвуют все. " if more else ". ")
    slow = {}
    for n in sizes:
        for p in procs:
            if s("block", n, p) < 1:
                slow.setdefault(p, []).append(n)
    if slow:
        worst = min((s("block", n, p), n, p) for p, ns in slow.items() for n in ns)
        text += ("Параллельная программа медленнее однопроцессной "
                 + "; ".join(f"при $P = {p}$ и {n_range(ns)}" for p, ns in sorted(slow.items()))
                 + f" (в худшем случае $S_{{\\text{{отн}}}}$ = {dec(worst[0], 3 if worst[0] < 0.1 else 2)} "
                   f"при $N = {worst[1]}$, $P = {worst[2]}$; п. «Большое число процессов»). ")
    seq_ratio = [ratio(m.t_seq(n), m.t("mpi", mode, n, 1)) for mode in MODES for n in sizes]
    lo, hi = min(seq_ratio), max(seq_ratio)
    text += (f"\n\nОтношение $T_{{\\text{{посл}}}} / T_1$ = {dec(lo)}–{dec(hi)}: "
             + ("программа MPI на одном процессе работает так же быстро, как последовательная (упаковка строки $j$ "
                "и рассылки при $P = 1$ дёшевы), поэтому абсолютное ускорение (рис. <<fig:speedup_n_abs.png>>) "
                "практически совпадает с относительным." if lo >= 0.97 else
                f"программа MPI на одном процессе медленнее последовательной до {percent(1 - lo)} (упаковка строки $j$ "
                "и рассылки), и на столько же абсолютное ускорение (рис. <<fig:speedup_n_abs.png>>) меньше "
                "относительного."))
    v["analysis_n"] = text

    # --- зависимость от P: модель и закон Амдала
    best_fixed = {n: max(procs, key=lambda p: s("block", n, p)) for n in fixed}
    text = (f"При {n_equals(fixed)} наибольшее относительное ускорение стратегии `block` — "
            f"{and_list(dec(s('block', n, best_fixed[n])) for n in fixed)} ("
            + (f"во всех случаях при $P = {best_fixed[fixed[0]]}$" if len(set(best_fixed.values())) == 1 else
               f"при $P$ = {and_list(str(best_fixed[n]) for n in fixed)} соответственно")
            + "; таблица <<tab:speedup_p>>, рис. <<fig:speedup_p.png>>).\n\n"
            "Чтобы отделить обмены от свойств самого распределения работы, построена модель без затрат на обмены: "
            "время операции над блоком пропорционально числу её операций, процессы одинаково быстрые; на шаге $j$ "
            "каждый процесс считает свои строки $i > j$ (их $\\lfloor m/P \\rfloor$ или $\\lceil m/P \\rceil$ из $m$), "
            "и шаг длится столько, сколько у самого нагруженного процесса, плюс последовательная часть. ")
    if p_agree:
        dev = max(abs(r - 1) for p in ratios if p <= p_agree for r in ratios[p])
        p_ex = 4 if 4 in within and 4 <= p_agree else p_agree
        text += (f"При $P \\leq {p_agree}$ замеры совпадают с моделью в пределах {percent(dev)} (таблица <<tab:model>>, "
                 f"рис. <<fig:efficiency_p.png>>): например, при $N = {big}$ и $P = {p_ex}$ модель даёт "
                 f"{dec(model['block', big, p_ex])}, замер — {dec(s('block', big, p_ex))}. ")
    else:
        text += "Уже при наименьшем $P$ замеры заметно ниже модели (таблица <<tab:model>>). "
    a_meas = [alpha["block", n] for n in fixed]
    a_mod = [alpha_model["block", n] for n in fixed]
    excess = [x - y for x, y in zip(a_meas, a_mod)]
    text += (f"\n\nПодгонка закона Амдала по $P \\leq {max(fit)}$ даёт для стратегии `block` "
             f"$\\alpha$ = {slashed(a_meas)} при $N = {' / '.join(map(str, fixed))}$ (таблица <<tab:amdahl>>), у модели — "
             f"{slashed(a_mod)}, оценка $1{{,}}5\\, b/N$ — {slashed(alpha_theory['block', n] for n in fixed)}. Разница "
             "между моделью и оценкой — неравномерность распределения (на последних шагах строк меньше, чем процессов), "
             "между замером и моделью — время рассылок и ожидания")
    text += (": она убывает с ростом $N$, как и предсказывает отношение вычислений к обменам $\\sim 2N/(3P)$. "
             if all(x > y for x, y in zip(excess, excess[1:])) else
             "; с ростом $N$ она в целом уменьшается, но не строго — сказывается разброс замеров. ")
    if wide:
        text += (f"Подгонка по всем $P \\leq {p_cores}$ (рис. <<fig:speedup_p.png>>) даёт большие значения — "
                 f"{slashed(alpha_cores['block', n] for n in fixed)}: их завышают замеры при $P > {max(fit)}$, "
                 "которые ниже модели. ")
    e_fit = [s("block", n, max(fit)) / max(fit) for n in fixed]
    text += f"Эффективность при $P = {max(fit)}$ — {dec(min(e_fit))}–{dec(max(e_fit))}."
    if p_agree and p_agree < p_cores:
        text += f" При $P > {p_agree}$ ускорение заметно ниже модели (п. «Большое число процессов»)."
    v["analysis_p"] = text

    # --- стратегии: разница меньше 2 % — «практически так же»
    def compare(p):
        r = {n: s("phases", n, p) / s("block", n, p) for n in sizes}
        return ([n for n in sizes if r[n] >= 1.02], [n for n in sizes if 0.98 < r[n] < 1.02],
                [n for n in sizes if r[n] <= 0.98])

    # основное сравнение — при наибольшем P, где замеры совпадают с моделью (устойчивы); при всех ядрах — с оговоркой
    p_main = p_agree if p_agree else p_cores
    win, same, lose = compare(p_main)
    gain = s("phases", big, p_main) / s("block", big, p_main) - 1

    def described(p, win_, same_, lose_, short=False):
        parts = [f"быстрее {n_set(win_, sizes)}"] if win_ else []
        parts += [f"практически так же (разница меньше 2 %) {n_set(same_, sizes)}"] if same_ else []
        parts += [f"медленнее {n_set(lose_, sizes)}"] if lose_ else []
        lead = f"При $P = {p}$ — " if short else f"При $P = {p}$ стратегия `phases` по сравнению с `block` работает "
        return lead + and_list(parts)

    text = (described(p_main, win, same, lose)
            + (f"; при $N = {big}$ выигрыш составляет {percent(gain)}" if gain >= 0.02 else "") + ". ")
    if p_main != p_cores:
        w2, s2, l2 = compare(p_cores)
        text += (described(p_cores, w2, s2, l2, short=True) + " (рис. <<fig:strategies.png>>); замеры при таком числе процессов "
                 "неустойчивы (п. «Большое число процессов»), и разница в обе стороны — в пределах разброса. ")
    else:
        text = text[:-2] + " (рис. <<fig:strategies.png>>). "
    text += ("У `phases` последовательна только факторизация блока $b \\times b$, а модификация диагонального блока "
             "идёт одновременно с остальными; подгонка закона Амдала даёт для неё меньшие $\\alpha$: "
             f"{slashed(alpha['phases', n] for n in fixed)} против {slashed(alpha['block', n] for n in fixed)} "
             "(таблица <<tab:amdahl>>). Цена — две рассылки на шаге вместо одной (тот же объём, но вдвое больше "
             "сообщений)")
    small_lose = [n for n in lose if n < 1000]
    text += (f", поэтому при малых $N$ ({n_set(small_lose, sizes).removeprefix('при ')}), где задержки рассылок заметны, "
             "она проигрывает." if small_lose else
             ": на этом компьютере (обмен через общую память) задержка сообщения мала, и выигрыш от меньшей "
             "последовательной части больше.")
    v["analysis_strategies"] = text

    # --- сравнение с OpenMP
    if has_omp:
        small = [p for p in omp_common if 1 < p <= (p_agree or 4)]
        t_ratio = [m.t("mpi", "block", big, p) / omp_times["block", big, p] for p in [1, *small]]
        n_mid = fixed[0]
        s_ratio = [s("block", n_mid, p) / s_omp("block", n_mid, p) for p in small if s_omp("block", n_mid, p)]
        p8 = 8 if 8 in omp_common else max(p for p in omp_common if p <= p_cores)
        r8 = m.t("mpi", "block", big, p8) / omp_times["block", big, p8]
        text = (f"При $N = {big}$ и $P \\leq {max(small or [1])}$ время программ MPI и OpenMP одинаково в пределах "
                f"{percent(max(abs(x - 1) for x in t_ratio))} (таблица <<tab:mpi_vs_omp>>, рис. <<fig:mpi_vs_omp.png>>): "
                "работа распределена одинаково (у процесса или потока — $\\lfloor m/P \\rfloor$ или "
                "$\\lceil m/P \\rceil$ блочных строк шага), и при большом $N$ обмены почти ничего не стоят. ")
        text += (f"При $P = {p8}$ программа MPI {'медленнее' if r8 > 1 else 'быстрее'} на {percent(abs(r8 - 1))}. "
                 if abs(r8 - 1) >= 0.02 else f"При $P = {p8}$ — тоже практически одинаково. ")
        if s_ratio:
            text += (f"При $N = {n_mid}$ ускорение MPI ниже на {dec(100 * (1 - max(s_ratio)), 0)}–"
                     f"{dec(100 * (1 - min(s_ratio)), 0)} % уже при малых $P$: пересылаемая строка $j$ копируется "
                     "в память каждого процесса, а процессы ждут рассылки, тогда как потоки OpenMP читают общую память "
                     "напрямую. " if max(s_ratio) < 0.98 else
                     f"При $N = {n_mid}$ и малых $P$ ускорения тоже близки. ")
        text += ("При большом числе процессов (потоков) обе программы упираются в одно и то же ограничение компьютера, "
                 "а замеры неустойчивы между прогонами, поэтому различия там в пределах разброса (п. «Большое число "
                 "процессов»).")
    else:
        text = ("Сравнение с OpenMP не выполнялось: результаты лабораторной работы № 3 получены на другом компьютере "
                "или отсутствуют.")
    v["analysis_omp"] = text

    # --- большое число процессов
    l3 = info.get("Кэш L3", "")
    text = ""
    gap = p_agree is not None and p_agree < p_cores
    if gap or p_agree is None:
        at_cores = [s(mode, n, p_cores) / model[mode, n, p_cores] for mode in MODES for n in fixed]
        text += (f"При $P = {p_cores}$ (все физические ядра) ускорение составляет {dec(100 * min(at_cores), 0)}–"
                 f"{dec(100 * max(at_cores), 0)} % от модели (таблица <<tab:model>>). ")
        plateau = [p for p in procs if 8 <= p < p_max] or [p for p in procs if p > (p_agree or 1)] or [p_max]
        gf_big = {p: max(gflops(big, m.t("mpi", mode, big, p)) for mode in MODES) for p in plateau}
        text += (f"Суммарная производительность при $N = {big}$ — "
                 + and_list(f"{dec(gf, 1)} GFLOP/s при $P = {p}$" for p, gf in gf_big.items())
                 + f" (на одном процессе — {dec(gflops(big, m.t('mpi', 'block', big, 1)), 1)} GFLOP/s)")
        if 8 in gf_big and p_cores in gf_big and p_cores > 8 and gf_big[p_cores] < 1.1 * gf_big[8]:
            text += (f": от 8 до {p_cores} процессов она почти не растёт, хотя ядер в {dec(p_cores / 8, 1)} раза "
                     "больше. ")
        else:
            text += ". "
        spread_low = max((m.median[k] / m.time[k] for k in m.time if k[0] == "mpi" and k[2] >= fixed[0]
                          and k[3] <= 4), default=1.0)
        spread_high = max((m.median[k] / m.time[k] for k in m.time
                           if k[0] == "mpi" and k[2] >= fixed[0] and (p_agree or 1) < k[3] < p_max), default=1.0)
        if spread_high > spread_low + 0.03:
            text += (f"Замеры при большом $P$ и менее устойчивы: медиана превышает минимальное время до "
                     f"{percent(spread_high - 1)}, а при $P \\leq 4$ — до {percent(spread_low - 1)}. ")
        text += ("\n\nРасхождение с моделью — не только обмены: модель не учитывает и свойства компьютера. ")
        if has_omp:
            text += ("Программа OpenMP (лабораторная работа № 3), у которой нет пересылок, упирается в то же "
                     "ограничение. ")
        text += ("Вероятные причины — снижение частоты процессора при нагрузке на много ядер и размещение процессов "
                 "операционной системой (два процесса на одном физическом ядре); после каждой рассылки процессы ждут "
                 f"друг друга, поэтому замедление одного задерживает весь шаг, а шагов много: $p = {blocks(big)}$ при "
                 f"$N = {big}$.")
    else:
        text += (f"При всех $P \\leq {p_cores}$ (до числа физических ядер) ускорение близко к модели: обмены и "
                 "ожидание почти ничего не стоят.")
    if p_max > p_cores:
        beyond = [p for p in procs if p > p_cores]
        s_c = s("block", big, p_cores)
        text += (f"\n\nПри $P > {p_cores}$ процессов больше, чем физических ядер, и два процесса делят одно ядро "
                 f"(Hyper-Threading): при $N = {big}$ ускорение "
                 + and_list(f"{dec(s('block', big, p))} при $P = {p}$" for p in beyond)
                 + f" против {dec(s_c)} при $P = {p_cores}$. ")
        collapse = [n for n in sizes if s("block", n, p_max) < 0.5]
        if collapse:
            slowdown = 1 / s("block", collapse[0], p_max)
            text += (f"При $P = {p_max}$ (все логические процессоры) и малых $N$ программа резко замедляется — "
                     f"{times_slower(slowdown)} при $N = {collapse[0]}$"
                     + (f", {times_slower(1 / s('block', collapse[-1], p_max))} при $N = {collapse[-1]}$"
                        if len(collapse) > 1 else "")
                     + " по сравнению с одним процессом. Вероятная причина — процессов столько же, сколько логических "
                       "процессоров, а реализация MPI ждёт сообщений активным опросом (процесс занимает свой "
                       "логический процессор и во время ожидания); когда операционной системе нужен процессор для своих "
                       "задач, она вытесняет какой-то процесс, и рассылка на каждом шаге ждёт его. ")
            if has_omp and s_omp("block", n0, p_max):
                text += (f"У программы OpenMP при $P = {p_max}$ такого обвала нет ({dec(s_omp('block', n0, p_max))} при "
                         f"$N = {n0}$): ожидающие потоки там засыпают. ")
            text += ("Практический вывод — запускать процессов MPI не больше, чем физических ядер.")
    v["analysis_many"] = text

    # --- выводы
    best_mode, best_n, best_pp = max(((mode, n, p) for mode in MODES for n in sizes for p in procs),
                                     key=lambda k: s(*k))
    conclusions = []
    conclusions.append(
        ("Распределение данных и обмены не изменили результат: при любом числе процессов и в обеих стратегиях "
         "фактор $L$ побитно совпадает с последовательным" if identical else
         "Погрешность параллельной программы — как у последовательной")
        + f", погрешность — {sci(e1)} при $N = {n1}$.")
    conclusions.append(
        "Программа MPI с циклическим распределением блочных строк и рассылкой строки $j$ (`MPI_Bcast`) даёт "
        f"относительное ускорение до {dec(s(best_mode, best_n, best_pp))} (стратегия `{best_mode}`, $N = {best_n}$, "
        f"$P = {best_pp}$); ускорение растёт с размерностью: вычисления растут как $N^3$, обмены — как $N^2$.")
    if p_agree:
        conclusions.append(
            f"До $P = {p_agree}$ ускорение близко к модели распределения строк без обменов; обмены увеличивают долю "
            f"последовательной работы $\\alpha$ (у `block` при $N = {fixed[0]}$ — {percent(alpha['block', fixed[0]])} "
            f"против {percent(alpha_model['block', fixed[0]])} у модели), и тем сильнее, чем меньше $N$.")
    if len(win) > len(sizes) / 2:
        conclusions.append("Стратегия `phases` (меньшая последовательная часть ценой второй рассылки) быстрее `block` "
                           "при большинстве $N$"
                           + (f" (на {percent(gain)} при $N = {big}$, $P = {p_main}$)." if gain >= 0.02 else "."))
    if has_omp:
        conclusions.append("При большом $N$ и малом $P$ программы MPI и OpenMP работают одинаково быстро; при меньших $N$ "
                           "MPI немного медленнее из-за пересылок.")
    if gap:
        conclusions.append(f"На всех {p_cores} ядрах ускорение ниже модели, процессы сверх числа ядер его почти не "
                           "увеличивают, а при всех логических процессорах программа с малыми $N$ резко замедляется: "
                           "при большом числе процессов ускорение ограничивает уже компьютер.")
    v["analysis_conclusions"] = "\n".join(f"{k}. {c}" for k, c in enumerate(conclusions, 1))

    # --- таблицы
    tables = {}
    tables["builds"] = ("Программы", ["Программа", "Ключи компиляции", "Назначение"],
                        [["cholesky_seq", m.flags.get("seq", "—"),
                          "последовательная (алгоритм этапа 2, без MPI): $T_{\\text{посл}}$"],
                         ["cholesky_mpi", m.flags.get("mpi", "—"), "параллельная (MPI): $T_1$ и $T_P$"]],
                        [2700, 3000, 3938], 99)
    tables["comm"] = (
        f"Объём рассылок и вычислений ($b = {b}$)",
        ["$N$", "$p$", "Рассылок, block", "Рассылок, phases", "Чисел", "Объём", "Операций на число"],
        comm_rows, [900, 800, 1500, 1500, 1938, 1300, 1700], 0)
    sweep_b = sorted({bb for n in sweep_n for bb in m.sweep[n]})
    tables["sweep"] = (
        "Подбор размера блока: время последовательной программы",
        ["$b$", *(f"$T$ при $N = {n}$, мс" for n in sweep_n)] if sweep_n else ["$b$", "Время"],
        [[f"**{bb}**" if bb == b else str(bb), *(ms(m.sweep[n][bb]) if bb in m.sweep[n] else "—" for n in sweep_n)]
         for bb in sweep_b] if sweep_n else [[str(b), "подбор не выполнялся"]],
        [1600, *([(TABLE_WIDTH - 1600) // max(1, len(sweep_n))] * max(1, len(sweep_n)))], 0)
    tables["errors"] = (
        "Погрешность разложения", ["$N$", "Погрешность $e(N)$"],
        [[str(n), sci(min(err_by_n[n])) if identical else f"{sci(min(err_by_n[n]))} … {sci(max(err_by_n[n]))}"]
         for n in sizes], [3000, 6638], 0)
    width = (TABLE_WIDTH - 1000 - 2 * 1400) // len(p_tab)
    for mode in MODES:
        tables[f"speedup_{mode}"] = (
            f"Время и относительное ускорение $S_{{\\text{{отн}}}}$, стратегия {mode} ($b = {b}$)",
            ["$N$", "$T_{\\text{посл}}$, мс", "$T_1$, мс", *(f"$P = {p}$" for p in p_tab)],
            [[str(n), ms(m.t_seq(n)), ms(m.t("mpi", mode, n, 1)),
              *(dec(s(mode, n, p), 3 if s(mode, n, p) < 0.1 else 2) for p in p_tab)] for n in sizes],
            [1000, 1400, 1400, *([width] * len(p_tab))], 0)
    width = (TABLE_WIDTH - 1000) // (2 * len(fixed))
    tables["speedup_p"] = (
        "Относительное ускорение $S$ и эффективность $E$ от числа процессов, стратегия block",
        ["$P$", *(f"{q}, $N = {n}$" for n in fixed for q in ("$S$", "$E$"))],
        [[str(p), *(x for n in fixed for x in (dec(s("block", n, p)), dec(s("block", n, p) / p)))] for p in procs],
        [1000, *([width] * (2 * len(fixed)))], 0)
    tables["model"] = (
        f"Относительное ускорение и модель распределения строк без обменов при $N = {big}$",
        ["$P$", "block: замер", "block: модель", "phases: замер", "phases: модель"],
        [[str(p), *(x for mode in MODES for x in (dec(s(mode, big, p)), dec(model[mode, big, p])))] for p in within],
        [1238, 2100, 2100, 2100, 2100], 0)
    tables["amdahl"] = (
        "Доля последовательной работы $\\alpha$ (закон Амдала)",
        ["Стратегия", "$N$", *([f"По замерам, $P \\leq {p_cores}$ (как на графике)"] if wide else []),
         f"По замерам, $P \\leq {max(fit)}$", "Модель без обменов", "Оценка"],
        [[mode, str(n), *([percent(alpha_cores[mode, n])] if wide else []), percent(alpha[mode, n]),
          percent(alpha_model[mode, n]),
          (f"$1{{,}}5\\, b/N$ = {percent(alpha_theory[mode, n])}" if mode == "block" else
           f"$b^2/N^2$ = {percent(alpha_theory[mode, n])}")] for mode in MODES for n in fixed],
        [1500, 900, 1900, 1700, 1400, 2238] if wide else [1500, 1100, 2400, 2000, 2638], 0)
    if has_omp:
        tables["mpi_vs_omp"] = (
            f"Относительное ускорение программ MPI и OpenMP (лабораторная работа № 3) при $N = {big}$",
            ["$P$", "MPI, block", "OpenMP, block", "MPI, phases", "OpenMP, phases"],
            [[str(p), dec(s("block", big, p)), dec(s_omp("block", big, p)), dec(s("phases", big, p)),
              dec(s_omp("phases", big, p))] for p in omp_common],
            [1238, 2100, 2100, 2100, 2100], 0)
    tables["hardware"] = (None, ["Параметр", "Значение"], hardware_rows(info, m), [4819, 4819], 99)
    return v, tables, {"omp": has_omp}


# ---------------------------------------------------------------- разметка шаблона
SPECIAL = re.compile(r"^(#{1,3} |!table |!figure |```|- |\d+\. )")


def parse_blocks(text: str) -> list:
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append(("code", lines[i + 1:j]))
            i = j + 1
        elif heading := re.match(r"(#{1,3}) (.*)", line):
            blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
            i += 1
        elif line.startswith("!table "):
            blocks.append(("table", line.split()[1]))
            i += 1
        elif line.startswith("!figure "):
            file, _, caption = line[len("!figure "):].partition("|")
            blocks.append(("figure", file.strip(), caption.strip()))
            i += 1
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            numbered = not line.startswith("- ")
            items = []
            while i < len(lines) and (re.match(r"\d+\. ", lines[i]) if numbered else lines[i].startswith("- ")):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i]).strip())
                i += 1
            blocks.append(("list", numbered, items))
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not SPECIAL.match(lines[i]):
                para.append(lines[i].strip())
                i += 1
            text_ = " ".join(para)
            if text_.startswith("$$") and text_.endswith("$$"):
                blocks.append(("display", text_[2:-2].strip()))
            else:
                blocks.append(("para", text_))
    return blocks


def substitute(text: str, values: dict) -> str:
    """<<имя>> → значение; <<fig:файл>> и <<tab:имя>> → номер рисунка/таблицы по порядку появления.
    Два прохода: в сгенерированных значениях тоже могут быть ссылки на рисунки и таблицы."""
    numbers, figures, tables = {}, 0, 0
    for line in text.splitlines():
        if line.startswith("!figure "):
            figures += 1
            numbers["fig:" + line[len("!figure "):].partition("|")[0].strip()] = str(figures)
        elif line.startswith("!table "):
            tables += 1
            numbers["tab:" + line.split()[1]] = str(tables)
    values = {**values, **numbers}
    for _ in range(2):
        missing = sorted(set(re.findall(r"<<([^<>]+)>>", text)) - set(values))
        if missing:
            sys.exit(f"В шаблоне неизвестные подстановки: {', '.join(missing)}")
        text = re.sub(r"<<([^<>]+)>>", lambda mt: values[mt.group(1)], text)
    return text


# ---------------------------------------------------------------- формулы
def fix_omml(xml: str) -> str:
    """Приводит OMML от pandoc к схеме Office: count раньше mcJc; nor и sty вместе не допускаются."""
    xml = re.sub(r"(<m:mcPr>)(<m:mcJc [^>]*/>)(<m:count [^>]*/>)", r"\1\3\2", xml)
    return re.sub(r"(<m:nor ?/>)<m:sty [^>]*/>", r"\1", xml)


def decimal_commas_as_text(tex: str) -> str:
    """«1{,}5» → «\\text{1,5}»: иначе pandoc делит число на «1», «,», «5», и Word может поставить пробел
    после запятой."""
    return re.sub(r"(\d+)\{,\}(\d+)", r"\\text{\1,\2}", tex)


def latex_to_omml(snippets: list) -> list:
    """[(tex, display)] → [OMML] одним вызовом pandoc."""
    if not snippets:
        return []
    md = "\n\n".join(f"$${decimal_commas_as_text(tex)}$$" if display else f"${decimal_commas_as_text(tex)}$"
                     for tex, display in snippets)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "math.docx"
        pypandoc.convert_text(md, "docx", format="markdown", outputfile=str(out))
        xml = zipfile.ZipFile(out).read("word/document.xml").decode("utf-8")
    paragraphs = re.findall(r"<w:p\b[^>]*>.*?</w:p>", xml, re.S)
    result = []
    for (tex, display), p in zip(snippets, paragraphs):
        found = re.search(r"<m:oMathPara>.*?</m:oMathPara>" if display else r"<m:oMath>.*?</m:oMath>", p, re.S)
        if not found:
            sys.exit(f"pandoc не смог перевести формулу: {tex}")
        result.append(fix_omml(found.group(0)))
    if len(result) != len(snippets):
        sys.exit("pandoc вернул не все формулы")
    return result


# ---------------------------------------------------------------- документ
class Renderer:
    """Блоки разметки → XML тела документа в стилях отчёта этапа 1."""

    def __init__(self, tables: dict, results_dir: Path, omml=None):
        self.tables, self.results_dir = tables, results_dir
        self.omml = iter(omml) if omml is not None else None
        self.snippets = []   # формулы в порядке появления (первый проход)
        self.images = []     # (имя файла, путь)
        self.lists = 0       # число нумерованных списков
        self.figure_no = self.table_no = 0

    def math(self, tex, display):
        if self.omml is None:
            self.snippets.append((tex, display))
            return ""
        return next(self.omml)

    def runs(self, text, rpr=""):
        out = []
        for part in re.split(r"(\$[^$]+\$|\*\*[^*]+\*\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("$"):
                out.append(self.math(part[1:-1].strip(), False))
            elif part.startswith("**"):
                out.append(run(part[2:-2], rpr + "<w:b/><w:bCs/>"))
            elif part.startswith("`"):
                out.append(run(part[1:-1], rpr + '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                                                  '<w:sz w:val="24"/><w:szCs w:val="24"/>'))
            else:
                out.append(run(part, rpr))
        return "".join(out)

    def paragraph(self, text, style="1-", ppr_extra="", rpr=""):
        return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr_extra}</w:pPr>{self.runs(text, rpr)}</w:p>'

    def table(self, name):
        caption, header, rows, widths, first_centered = self.tables[name]
        xml = ""
        if caption:
            self.table_no += 1
            xml += self.paragraph(f"Таблица {self.table_no}. {caption}", "-1")
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

        def row(cells, head):
            tcs = []
            for col, (text, width) in enumerate(zip(cells, widths)):
                shading = '<w:shd w:val="clear" w:color="auto" w:fill="DDEBF7"/>' if head else ""
                jc = '<w:jc w:val="center"/>' if head or col >= first_centered else ""
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shading}<w:vAlign w:val="center"/>'
                           f'</w:tcPr>{self.paragraph(text, "aff1", jc)}</w:tc>')
            trpr = "<w:trPr><w:tblHeader/></w:trPr>" if head else ""
            return f"<w:tr>{trpr}{''.join(tcs)}</w:tr>"

        borders = "".join(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
                          for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
        xml += (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
                f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="57" w:type="dxa"/>'
                f'<w:right w:w="57" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="04A0"/></w:tblPr>'
                f'<w:tblGrid>{grid}</w:tblGrid>{row(header, True)}{"".join(row(r, False) for r in rows)}</w:tbl>')
        return xml + '<w:p><w:pPr><w:pStyle w:val="-1"/></w:pPr></w:p>'

    def figure(self, file, caption):
        path = self.results_dir / file
        with open(path, "rb") as f:
            width, height = struct.unpack(">II", f.read(24)[16:24])  # заголовок PNG (IHDR)
        self.figure_no += 1
        self.images.append((f"lab{LAB_NUMBER}_{file}", path))
        rid, cx = f"rIdLabFig{self.figure_no}", IMAGE_WIDTH_EMU
        cy = cx * height // width
        pid = 1000 + self.figure_no
        a = "http://schemas.openxmlformats.org/drawingml/2006/main"
        pic = "http://schemas.openxmlformats.org/drawingml/2006/picture"
        drawing = (
            f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{pid}" name="Рисунок {self.figure_no}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{a}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="{a}"><a:graphicData uri="{pic}"><pic:pic xmlns:pic="{pic}"><pic:nvPicPr>'
            f'<pic:cNvPr id="{pid}" name="{escape(file)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
            f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr>'
            f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
            f'</a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
        image = f'<w:p><w:pPr><w:pStyle w:val="-"/><w:keepNext/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>{drawing}</w:r></w:p>'
        return image + self.paragraph(f"Рисунок {self.figure_no} – {caption}", "-", rpr="<w:i/><w:iCs/>")

    def code(self, lines):
        rpr = ('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/><w:szCs w:val="18"/>'
               '<w:lang w:val="en-US"/>')
        border = "".join(f'<w:{side} w:val="single" w:sz="2" w:space="0" w:color="CCCCCC"/>'
                         for side in ("top", "left", "bottom", "right"))
        body = "<w:r><w:rPr>" + rpr + "</w:rPr><w:br/></w:r>"
        return (f'<w:p><w:pPr><w:pBdr>{border}</w:pBdr><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>'
                f'<w:spacing w:before="100" w:after="200"/></w:pPr>'
                + body.join(run(line, rpr) for line in lines) + "</w:p>")

    def render(self, blocks):
        out = []
        for block in blocks:
            kind = block[0]
            if kind == "heading":
                out.append(f'<w:p><w:pPr><w:pStyle w:val="{block[1]}"/></w:pPr>{run(block[2])}</w:p>')
            elif kind == "para":
                out.append(self.paragraph(block[1]))
            elif kind == "display":
                out.append(f'<w:p><w:pPr><w:pStyle w:val="1-"/></w:pPr>{self.math(block[1], True)}</w:p>')
            elif kind == "list":
                num_id = BULLET_NUM_ID
                if block[1]:
                    self.lists += 1
                    num_id = str(1000 + self.lists)
                numpr = f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
                out.extend(self.paragraph(item, "1-", numpr) for item in block[2])
            elif kind == "table":
                out.append(self.table(block[1]))
            elif kind == "figure":
                out.append(self.figure(block[1], block[2]))
            elif kind == "code":
                out.append(self.code(block[1]))
        return "".join(out)


def run(text, rpr=""):
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def top_level_elements(body: str) -> list:
    return re.findall(r"<w:p\b[^>]*>.*?</w:p>|<w:tbl>.*?</w:tbl>|<w:sectPr\b.*?</w:sectPr>", body, re.S)


def build(results_dir: Path = RESULTS_DIR, output: Path = OUTPUT, omp_dir: Path = OPENMP_RESULTS):
    m = Measurements(results_dir / "measurements.csv")
    info = read_platform(results_dir / "platform.txt")
    if not (results_dir / "scheme.png").exists():  # схема от замеров не зависит; её рисует и plot_speedup.py
        draw_scheme(results_dir)
    values, tables, flags = compute(m, info, results_dir, omp_dir)
    text = "\n".join(line for line in TEMPLATE_TEXT.read_text(encoding="utf-8").splitlines()
                     if not line.startswith("%"))  # комментарии шаблона
    blocks = parse_blocks(substitute(apply_conditions(text, flags), values))

    first_pass = Renderer(tables, results_dir)
    first_pass.render(blocks)
    hardware_pass = Renderer(tables, results_dir)
    hardware_pass.table("hardware")
    renderer = Renderer(tables, results_dir, latex_to_omml(first_pass.snippets + hardware_pass.snippets))
    body = renderer.render(blocks)
    hardware = renderer.table("hardware")

    template = zipfile.ZipFile(TEMPLATE_DOCX)
    document = template.read("word/document.xml").decode("utf-8")
    head, _, rest = document.partition("<w:body>")
    items = top_level_elements(rest.partition("</w:body>")[0])
    title_end = next(k for k, it in enumerate(items) if "Самара" in "".join(re.findall(r"<w:t[^>]*>([^<]*)", it)))
    title = items[:title_end + 1]
    title = [re.sub(r"(ЛАБОРАТОРНАЯ РАБОТА № </w:t></w:r>.*?<w:t[^>]*>)1(</w:t>)", rf"\g<1>{LAB_NUMBER}\2", it, flags=re.S)
             for it in title]
    lab_heading = items[title_end + 1].replace("№ 1", f"№ {LAB_NUMBER}")
    variant_table, page_break, tech_heading, tech_intro = items[title_end + 2:title_end + 6]
    spacer = '<w:p><w:pPr><w:pStyle w:val="1-"/><w:ind w:firstLine="0"/></w:pPr></w:p>'
    new_body = "".join([*title, lab_heading, variant_table, page_break, tech_heading, tech_intro, hardware, spacer,
                        body, items[-1]])
    document = f"{head}<w:body>{new_body}</w:body></w:document>"

    numbering = template.read("word/numbering.xml").decode("utf-8")
    nums = "".join(f'<w:num w:numId="{1000 + k}"><w:abstractNumId w:val="{DECIMAL_ABSTRACT_NUM}"/>'
                   f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>'
                   for k in range(1, renderer.lists + 1))
    anchor = "<w:numIdMacAtCleanup" if "<w:numIdMacAtCleanup" in numbering else "</w:numbering>"
    numbering = numbering.replace(anchor, nums + anchor, 1)

    rels = template.read("word/_rels/document.xml.rels").decode("utf-8")
    old_images = re.findall(r'<Relationship [^>]*Type="[^"]*/image"[^>]*/>', rels)
    old_media = {"word/" + re.search(r'Target="([^"]+)"', r).group(1) for r in old_images}
    for r in old_images:
        rels = rels.replace(r, "")
    rels = rels.replace("</Relationships>", "".join(
        f'<Relationship Id="rIdLabFig{k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        f'image" Target="media/{name}"/>' for k, (name, _) in enumerate(renderer.images, 1)) + "</Relationships>")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as out:
        for entry in template.infolist():
            if entry.filename in old_media:
                continue
            data = {"word/document.xml": document, "word/numbering.xml": numbering,
                    "word/_rels/document.xml.rels": rels}.get(entry.filename)
            out.writestr(entry, data.encode("utf-8") if data is not None else template.read(entry.filename))
        for name, path in renderer.images:
            out.write(path, f"word/media/{name}")
    print(f"Отчёт сохранён: {output}")


if __name__ == "__main__":
    build()
