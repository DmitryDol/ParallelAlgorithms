"""
Этап 3: черновик отчёта «Отчёт_ПП_3_Абрамкин_Должиков.docx» из результатов замеров.

Запуск из корня репозитория (после bench.py и plot_speedup.py; run_all.py запускает его сам):
    uv run python lab3_openmp/report/make_report.py

Оформление берётся из отчёта этапа 1: титульный лист (меняется только номер работы), стили, колонтитулы,
вид таблиц, рисунков и листингов. Текст — из report_template.md (его можно править), числа, таблицы и графики —
из lab3_openmp/results/ (measurements.csv, platform.txt, autopar_report.txt, *.png). Формулы LaTeX переводятся
в формулы Word программой pandoc (пакет pypandoc-binary). Разделы «Анализ результатов» собираются из чисел и
учитывают компилятор (MSVC или GCC); после нового прогона их нужно перечитать.
Разметка шаблона, формулы и вёрстка документа — копия генератора этапа 2.
"""

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
from plot_speedup import (Measurements, amdahl_alpha, physical_cores, ratio, read_platform,  # noqa: E402
                          static_schedule_speedup)
from programs import RESULTS_DIR  # noqa: E402

TEMPLATE_DOCX = HERE.parent.parent / "lab1_python" / "report" / "Отчёт_ПП_1_Абрамкин_Должиков.docx"
TEMPLATE_TEXT = HERE / "report_template.md"
OUTPUT = HERE / "Отчёт_ПП_3_Абрамкин_Должиков.docx"
LAB_NUMBER = 3

TABLE_WIDTH = 9638           # ширина таблиц (twips), как в отчёте этапа 1
IMAGE_WIDTH_EMU = 5760000    # ширина рисунков — 16 см
BULLET_NUM_ID = "33"         # маркированный список шаблона
DECIMAL_ABSTRACT_NUM = "0"   # нумерованный список шаблона («1.», «2.»)
MACHINE_EPSILON = 2.220446049250313e-16
MODES = ("block", "phases")
AGREEMENT = 0.9              # замер «совпадает с моделью», если не ниже 90 % от неё
OUTSIDE_TIMING = {"make_test_matrix", "max_error_vs_exact"}  # функции cholesky.cpp вне замера времени
# значение _OPENMP → версия стандарта
OPENMP_VERSIONS = {"200203": "2.0", "200505": "2.5", "200805": "3.0", "201107": "3.1", "201307": "4.0",
                   "201511": "4.5", "201811": "5.0", "202011": "5.1", "202111": "5.2", "202411": "6.0"}


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
    if ns == sizes[sizes.index(ns[0]):]:
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


def compiler_family(compiler: str) -> str:
    return "msvc" if compiler.startswith("MSVC") else "gcc"


def openmp_text(value: str) -> str:
    if not value:
        return "—"
    return f"OpenMP {OPENMP_VERSIONS[value]}" if value in OPENMP_VERSIONS else f"OpenMP (_OPENMP = {value})"


def table_threads(threads, cores):
    """Числа потоков для таблиц: 2, 4, 8, число ядер и наибольшее — те, что были в замерах."""
    wanted = {2, 4, 8, max(threads)} | ({cores} if cores else set())
    return [p for p in threads if p > 1 and p in wanted]


def agreement_limit(ratios: dict):
    """Наибольшее P, до которого (включительно) все отношения «замер / модель» не ниже AGREEMENT; None — если
    уже при наименьшем P ниже."""
    limit = None
    for p in sorted(ratios):
        if min(ratios[p]) < AGREEMENT:
            break
        limit = p
    return limit


def parse_autopar(text: str) -> dict:
    """Разбор results/autopar_report.txt (его пишет bench.py): MSVC сообщает о каждом цикле (C5011 — распараллелен,
    C5012 — нет, с кодом причины), GCC — только о распараллеленных; причины отказа — сводкой «N × причина»."""
    result = {"found": not text.startswith("Отчёт компилятора об автопараллелизации не найден"),
              "loops": [], "parallelized": [], "reasons": []}
    for line in text.splitlines():
        if m := re.match(r"cholesky\.cpp:(\d+) \(функция ([\w?]+)\): (C501[12]): (.*)$", line):
            number, function = int(m.group(1)), m.group(2)
            code = re.search(r"(\d{3,4})\D*$", m.group(4)) if m.group(3) == "C5012" else None
            result["loops"].append((number, function, code.group(1) if code else None))
            if m.group(3) == "C5011":
                result["parallelized"].append((number, function))
        elif m := re.match(r"cholesky\.cpp:(\d+) \(функция ([\w?]+)\): parallelizing ", line):
            result["parallelized"].append((int(m.group(1)), m.group(2)))
        elif m := re.match(r"\s+(\d+) × (.*)$", line):
            result["reasons"].append((int(m.group(1)), m.group(2).strip()))
    result["compiler_lines"] = bool(result["loops"])
    return result


# ---------------------------------------------------------------- данные
def hardware_rows(info, m):
    rows = [[key, info[key]] for key in ("Процессор", "Частота процессора", "Физических ядер", "Логических процессоров")
            if key in info]
    rows += [[key, value] for key, value in info.items() if key.startswith("Кэш")]
    compiler = m.compiler.get("omp") or m.compiler.get("seq", "—")
    key = "/openmp" if compiler_family(compiler) == "msvc" else "-fopenmp"
    rows += [["Операционная система", info.get("ОС", "—")],
             ["Компилятор", compiler],
             ["Реализация OpenMP", f"{openmp_text(m.openmp)}, ключ {key}"],
             ["Язык", "C++ (стандарт C++17)"]]
    return rows


def compute(m: Measurements, info: dict, autopar: dict) -> tuple[dict, dict]:
    """Подстановки для шаблона и таблицы."""
    compiler = m.compiler.get("omp") or m.compiler.get("seq", "—")
    family = compiler_family(compiler)
    cores = physical_cores(info)
    sizes, threads, b = m.sizes, m.threads, m.block
    n0, n1 = sizes[0], sizes[-1]
    fixed = m.fixed_sizes()
    big = fixed[-1]
    within = [p for p in threads if cores is None or p <= cores]  # не больше, чем физических ядер
    p_cores, p_max = max(within), max(threads)
    p_tab = table_threads(threads, cores)
    blocks = lambda n: -(-n // b)  # noqa: E731 — число блочных строк

    def s(mode, n, p):
        return m.s_rel(mode, n, p)

    def gflops(n, t):
        return n**3 / 3 / t / 1e9

    model = {(mode, n, p): static_schedule_speedup(n, b, p, mode) for mode in MODES for n in fixed for p in within}
    ratios = {p: [s(mode, n, p) / model[mode, n, p] for mode in MODES for n in fixed] for p in within if p > 1}
    p_agree = agreement_limit(ratios)
    fit = [p for p in within if p_agree is None or p <= p_agree]
    alpha = {(mode, n): amdahl_alpha(fit, [s(mode, n, p) for p in fit]) for mode in MODES for n in fixed}
    alpha_model = {(mode, n): amdahl_alpha(fit, [model[mode, n, p] for p in fit]) for mode in MODES for n in fixed}
    alpha_theory = {("block", n): 1.5 * b / n for n in fixed} | {("phases", n): (b / n) ** 2 for n in fixed}
    wide = fit != within  # подгонка до числа ядер (график speedup_p) отличается от подгонки по совпадению с моделью
    alpha_cores = {(mode, n): amdahl_alpha(within, [s(mode, n, p) for p in within]) for mode in MODES for n in fixed}

    v = {
        "compiler": compiler,
        "openmp": openmp_text(m.openmp),
        "repeats": str(m.repeats),
        "sizes": n_list(sizes),
        "threads": n_list(threads),
        "p_cores": str(p_cores),
        "fixed_text": n_equals(fixed),
    }
    autopar_p = and_list(map(str, m.autopar_threads)) if m.autopar_threads else "—"  # после «$P$ = »

    # --- ключи компиляции и вариант Б (разные у MSVC и GCC/Clang)
    if family == "msvc":
        v["autopar_flags"] = "ключ компилятора MSVC `/Qpar`"
        v["autopar_method"] = (
            "Программа `cholesky_autopar` компилируется без OpenMP (директивы `#pragma omp` при этом не действуют) "
            "с ключом `/Qpar`: компилятор распараллеливает циклы, независимость итераций которых может доказать сам, "
            "и выполняет их средствами своей среды OpenMP; число потоков задаётся переменной окружения "
            "`OMP_NUM_THREADS`. Ключ `/Qpar-report:2` выводит для каждого цикла сообщение C5011 «цикл распараллелен» "
            f"или C5012 «цикл не распараллелен» с кодом причины. Замеры выполнены при $P$ = {autopar_p}.")
        v["flags_explanation"] = (
            "Ключ `/openmp` включает директивы OpenMP (CMake передаёт его как `-openmp` — это тот же ключ). MSVC "
            "поддерживает стандарт OpenMP 2.0, в котором индекс параллельного цикла должен быть целым числом со знаком; "
            "наши циклы этому условию удовлетворяют. В сборках без OpenMP директивы `#pragma omp` не действуют, "
            "предупреждение о них отключено (`/wd4068`).")
    else:
        v["autopar_flags"] = "ключ компилятора GCC `-ftree-parallelize-loops`"
        v["autopar_method"] = (
            "Программы `cholesky_autopar_p<P>` компилируются без OpenMP (директивы `#pragma omp` при этом не "
            "действуют) с ключом `-ftree-parallelize-loops=P`: компилятор распараллеливает циклы, независимость "
            "итераций которых может доказать сам. Число потоков задаётся при компиляции, поэтому для каждого $P$ "
            f"собрана своя программа; замеры выполнены при $P$ = {autopar_p}. Распараллеленные циклы сообщает ключ "
            "`-fopt-info-loop`, причины отказа — дамп прохода `parloops`.")
        v["flags_explanation"] = (
            f"Ключ `-fopenmp` включает директивы OpenMP ({openmp_text(m.openmp)}). В сборках без OpenMP директивы "
            "`#pragma omp` не действуют, предупреждение о них отключено (`-Wno-unknown-pragmas`).")

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
    v["errors_note"] = (
        "При каждом $N$ погрешность одна и та же во всех сборках, при всех числах потоков и в обеих стратегиях — "
        "до последнего разряда: параллельно выполняются независимые блоки, а порядок операций внутри блока "
        "не меняется." if identical else
        "Погрешность в разных сборках и при разном числе потоков различается только в последних разрядах "
        "(в таблице — наименьшее и наибольшее значения).")
    growth = "растёт с увеличением $N$" if e1 > e0 else "почти не зависит от $N$"
    v["analysis_correctness"] = (
        f"Во всём диапазоне $N = {n0} \\ldots {n1}$ погрешность составляет от {sci(e0)} до {sci(e1)} и, как на "
        f"этапах 1 и 2, {growth} из-за накопления ошибок округления в скалярных произведениях; наибольшая — не более "
        f"{round(e_max / MACHINE_EPSILON)} машинных эпсилон. "
        + ("Распараллеливание не изменило ни одного разряда результата: погрешность одна и та же при любом числе "
           "потоков, в обеих стратегиях и в обоих вариантах, а тесты подтверждают побитное совпадение фактора $L$. "
           if identical else "")
        + "Ускорение получено без потери точности.")

    # --- зависимость от N
    s_ref = {n: s("block", n, p_cores) for n in sizes}
    best_p = {n: max(threads, key=lambda p: s("block", n, p)) for n in sizes}
    groups = []
    for n in sizes:
        if groups and groups[-1][0] == best_p[n]:
            groups[-1][1].append(n)
        else:
            groups.append((best_p[n], [n]))
    n_best = max(sizes, key=lambda n: s("block", n, best_p[n]))
    p0 = blocks(n0)
    more = [p for p in threads if p > p0]
    text = (f"При $P = {p_cores}$ относительное ускорение стратегии `block` "
            f"{trend_text(s_ref[n0], n0, s_ref[n1], n1)} (таблица <<tab:speedup_block>>, рис. <<fig:speedup_n.png>>). "
            + ("Наибольшее ускорение достигается " + ", ".join(f"при {n_range(ns)} — при $P = {p}$" for p, ns in groups)
               + ". " if len(groups) > 1 else f"Наибольшее ускорение при всех $N$ достигается при $P = {groups[0][0]}$. ")
            + f"Самое большое значение — {dec(s('block', n_best, best_p[n_best]))} при $N = {n_best}$, "
              f"$P = {best_p[n_best]}$. С ростом $N$ работа одного шага $j$ растёт пропорционально $N^2 b$, а накладные "
              "расходы на запуск параллельного цикла и барьер в его конце от $N$ не зависят. Кроме того, при малых $N$ "
              f"блоков мало: при $N = {n0}$ блочных строк всего $p = {p0}$")
    text += (f", меньше, чем потоков при $P \\geq {more[0]}$, и часть потоков простаивает. " if more else ". ")
    slow = sorted(((s("block", n, p), n, p) for n in sizes for p in threads if s("block", n, p) < 1))
    if slow:
        worst = f"$S_{{\\text{{отн}}}}$ = {dec(slow[0][0])}"
        text += ("При " + and_list(f"$N = {n}$, $P = {p}$" for _, n, p in slow[:3])
                 + (f" и ещё в {len(slow) - 3} случаях" if len(slow) > 3 else "")
                 + " параллельная программа даже медленнее однопоточной ("
                 + (f"в худшем случае {worst}" if len(slow) > 1 else worst) + "). ")
    seq_ratio = [ratio(m.t_seq(n), m.t("omp", mode, n, 1)) for mode in MODES for n in sizes]
    lo, hi = min(seq_ratio), max(seq_ratio)
    text += (f"\n\nОтношение $T_{{\\text{{посл}}}} / T_1$ = {dec(lo)}–{dec(hi)}: "
             + ("программа OpenMP на одном потоке работает так же быстро, как последовательная, поэтому абсолютное "
                f"ускорение (рис. <<fig:speedup_n_abs.png>>) практически совпадает с относительным."
                if lo >= 0.97 else
                f"программа OpenMP на одном потоке медленнее последовательной до {percent(1 - lo)}, и на столько же "
                "абсолютное ускорение (рис. <<fig:speedup_n_abs.png>>) меньше относительного."))
    v["analysis_n"] = text

    # --- зависимость от P: модель и закон Амдала
    best_fixed = {n: max(threads, key=lambda p: s("block", n, p)) for n in fixed}
    text = (f"При {n_equals(fixed)} наибольшее относительное ускорение стратегии `block` — "
            f"{and_list(dec(s('block', n, best_fixed[n])) for n in fixed)} ("
            + (f"во всех случаях при $P = {best_fixed[fixed[0]]}$" if len(set(best_fixed.values())) == 1 else
               f"при $P$ = {and_list(str(best_fixed[n]) for n in fixed)} соответственно")
            + "; таблица <<tab:speedup_p>>, рис. <<fig:speedup_p.png>>).\n\n"
            "Чтобы отделить свойства алгоритма от свойств компьютера, построена модель распределения блоков: время "
            "операции над блоком пропорционально числу её операций, потоки одинаково быстрые, накладных расходов нет; "
            "на каждом шаге $j$ последовательная часть выполняется одним потоком, а параллельный цикл длится столько, "
            "сколько самая большая порция из $\\lceil m/P \\rceil$ блоков. ")
    if p_agree:
        dev = max(abs(r - 1) for p in ratios if p <= p_agree for r in ratios[p])
        p_ex = 4 if 4 in within and 4 <= p_agree else p_agree
        text += (f"При $P \\leq {p_agree}$ замеры совпадают с моделью в пределах {percent(dev)} (таблица "
                 f"<<tab:model>>, рис. <<fig:efficiency_p.png>>): например, при $N = {big}$ и $P = {p_ex}$ модель даёт "
                 f"{dec(model['block', big, p_ex])}, замер — {dec(s('block', big, p_ex))}. Значит, при таком числе "
                 "потоков ускорение ограничивают только свойства алгоритма: последовательная часть шага и "
                 "неравномерность распределения блоков.")
    else:
        text += "Уже при наименьшем $P$ замеры заметно ниже модели (таблица <<tab:model>>)."
    a_meas = [alpha["block", n] for n in fixed]
    text += (f"\n\nПодгонка закона Амдала по $P \\leq {max(fit)}$ даёт для стратегии `block` "
             f"$\\alpha$ = {slashed(a_meas)} при "
             f"$N = {' / '.join(map(str, fixed))}$ (таблица <<tab:amdahl>>); у модели — "
             f"{slashed(alpha_model['block', n] for n in fixed)}. Это больше оценки "
             f"$1{{,}}5\\, b/N$ = {slashed(alpha_theory['block', n] for n in fixed)}: "
             "к последовательной части добавляется неравномерность распределения — на последних шагах блоков "
             "меньше, чем потоков, а порция $\\lceil m/P \\rceil$ округляется вверх. ")
    text += ("Как и предсказывает оценка, $\\alpha$ убывает с ростом $N$. "
             if all(x > y for x, y in zip(a_meas, a_meas[1:])) else
             "Оценка предсказывает убывание $\\alpha$ с ростом $N$; по замерам оно не строгое — сказывается разброс. ")
    if wide:
        text += (f"Подгонка по всем $P \\leq {p_cores}$ (рис. <<fig:speedup_p.png>>) даёт большие значения — "
                 f"{slashed(alpha_cores['block', n] for n in fixed)}: их завышают замеры при $P > {max(fit)}$, "
                 "которые ниже модели. ")
    e_fit = [s("block", n, max(fit)) / max(fit) for n in fixed]
    text += f"Эффективность при $P = {max(fit)}$ — {dec(min(e_fit))}–{dec(max(e_fit))}."
    if p_agree and p_agree < p_cores:
        text += f" При $P > {p_agree}$ ускорение заметно ниже модели (п. «Большое число потоков»)."
    v["analysis_p"] = text

    # --- стратегии: разница меньше 2 % — «практически так же»
    def compare(p):
        r = {n: s("phases", n, p) / s("block", n, p) for n in sizes}
        return ([n for n in sizes if r[n] >= 1.02], [n for n in sizes if 0.98 < r[n] < 1.02],
                [n for n in sizes if r[n] <= 0.98])

    win, same, lose = compare(p_cores)
    gain = s("phases", big, p_cores) / s("block", big, p_cores) - 1
    parts = [f"быстрее {n_set(win, sizes)}"] if win else []
    parts += [f"практически так же (разница меньше 2 %) {n_set(same, sizes)}"] if same else []
    parts += [f"медленнее {n_set(lose, sizes)}"] if lose else []
    text = (f"При $P = {p_cores}$ стратегия `phases` по сравнению с `block` работает " + and_list(parts)
            + " (рис. <<fig:strategies.png>>)"
            + (f"; при $N = {big}$ выигрыш составляет {percent(gain)}" if gain >= 0.02 else "") + ". ")
    if p_agree and p_agree != p_cores:
        w2, _, _ = compare(p_agree)
        text += f"При $P = {p_agree}$ `phases` быстрее {n_set(w2, sizes)}. "
    text += ("У `phases` последовательна только факторизация блока $b \\times b$ ($\\alpha \\approx b^2/N^2$ вместо "
             "$1{,}5\\, b/N$), и подгонка закона Амдала даёт для неё меньшие $\\alpha$: "
             f"{slashed(alpha['phases', n] for n in fixed)} против "
             f"{slashed(alpha['block', n] for n in fixed)} (таблица <<tab:amdahl>>). "
             "Зато на каждом шаге у неё два параллельных цикла и два барьера вместо одного")
    small_lose = [n for n in lose if n < 1000]
    text += (f", поэтому при малых $N$ ({n_set(small_lose, sizes).removeprefix('при ')}), где накладные расходы "
             "на запуск циклов заметны, она может проигрывать." if small_lose else ".")
    v["analysis_strategies"] = text

    # --- большое число потоков
    l3 = info.get("Кэш L3", "")
    text = ""
    gap = p_agree is not None and p_agree < p_cores
    if gap or p_agree is None:
        at_cores = [s(mode, n, p_cores) / model[mode, n, p_cores] for mode in MODES for n in fixed]
        text += (f"При $P = {p_cores}$ (все физические ядра) ускорение составляет {dec(100 * min(at_cores), 0)}–"
                 f"{dec(100 * max(at_cores), 0)} % от модели (таблица <<tab:model>>). ")
        omp_gf = {(mode, n, p): gflops(n, m.t("omp", mode, n, p)) for mode in MODES for n in sizes for p in threads}
        top = max(omp_gf, key=omp_gf.get)
        one = [gflops(n, m.t("omp", "block", n, 1)) for n in fixed]
        text += (f"Наибольшая производительность — {dec(omp_gf[top], 1)} GFLOP/s (стратегия `{top[0]}`, $N = {top[1]}$, "
                 f"$P = {top[2]}$), на одном потоке — {dec(min(one), 1)}–{dec(max(one), 1)} GFLOP/s. ")
        # разброс «медиана / минимум» на тех же N, что в таблице S(P)
        spread_low = max((m.median[k] / m.time[k] for k in m.time if k[0] == "omp" and k[2] >= fixed[0]
                          and k[3] <= 4), default=1.0)
        spread_high = max((m.median[k] / m.time[k] for k in m.time
                           if k[0] == "omp" and k[2] >= fixed[0] and k[3] > (p_agree or 1)), default=1.0)
        if spread_high > spread_low + 0.03:
            text += (f"Замеры при большом $P$ и менее устойчивы: медиана превышает минимальное время до "
                     f"{percent(spread_high - 1)}, а при $P \\leq 4$ — до {percent(spread_low - 1)}. ")
        text += "\n\nРасхождение с моделью — не свойство алгоритма: модель его учитывает. "
        l3_kb = re.match(r"([\d,]+)\s*(КБ|МБ)", l3)
        n_small = fixed[0]
        l3_mb = float(l3_kb.group(1).replace(",", ".")) / (1024 if l3_kb.group(2) == "КБ" else 1) if l3_kb else None
        l_mb = 8 * n_small**2 / 2**20  # матрица L
        if l3_mb and 4 <= l_mb < l3_mb and min(s(mode, n_small, p_cores) / model[mode, n_small, p_cores]
                                               for mode in MODES) < AGREEMENT:
            text += (f"Это и не нехватка пропускной способности памяти: при $N = {n_small}$ матрица $L$ "
                     f"({dec(l_mb, 0)} МБ) целиком помещается в кэш L3 ({l3}), а ускорение так же "
                     "ниже модели. ")
        text += ("Вероятные причины — снижение частоты процессора при нагрузке на много ядер и размещение потоков "
                 "операционной системой: два потока на одном физическом ядре или поток на ядре, занятом другой "
                 "программой. После каждого шага $j$ все потоки ждут друг друга на барьере, поэтому замедление одного "
                 f"потока задерживает весь шаг, а шагов много: $p = {blocks(big)}$ при $N = {big}$.")
    else:
        text += (f"При всех $P \\leq {p_cores}$ (до числа физических ядер) ускорение близко к модели, то есть "
                 "ограничено только свойствами алгоритма.")
    if p_max > p_cores:
        s_c, s_m = s("block", big, p_cores), s("block", big, p_max)
        text += (f"\n\nПри $P > {p_cores}$ потоков больше, чем физических ядер, и два потока делят одно ядро "
                 "(Hyper-Threading). Ускорение при этом "
                 + (f"не растёт: при $N = {big}$ и $P = {p_max}$ оно равно {dec(s_m)} против {dec(s_c)} при "
                    f"$P = {p_cores}$" if s_m <= 1.05 * s_c else
                    f"растёт лишь до {dec(s_m)} при $N = {big}$, $P = {p_max}$ (против {dec(s_c)} при $P = {p_cores}$)")
                 + f"; при малых $N$ оно падает: при $N = {n0}$, $P = {p_max}$ — {dec(s('block', n0, p_max))}. "
                 "Второй поток на том же ядре делит с первым его вычислительные устройства и кэш, а накладные "
                 "расходы на синхронизацию растут с числом потоков.")
    v["analysis_many"] = text

    # --- автопараллелизация
    s_auto = [m.s_autopar(n, p) for n in sizes for p in m.autopar_threads if m.s_autopar(n, p)]
    kernel = [(line, f) for line, f in autopar["parallelized"] if f not in OUTSIDE_TIMING]
    outside = [(line, f) for line, f in autopar["parallelized"] if f in OUTSIDE_TIMING]
    text = ""
    if s_auto:
        text += (f"Программа, распараллеленная компилятором, ускорения не даёт: $S_{{\\text{{абс}}}}$ = "
                 f"{dec(min(s_auto))}–{dec(max(s_auto))} при $P$ = {autopar_p} "
                 if max(s_auto) < 1.1 else
                 f"Программа, распараллеленная компилятором, ускоряется до {dec(max(s_auto))} раза при "
                 f"$P$ = {autopar_p} ")
        text += "(таблица <<tab:autopar>>, рис. <<fig:autopar.png>>). "
    if not autopar["found"]:
        text += "Отчёт компилятора об автопараллелизации в результатах не найден. "
    else:
        if autopar["compiler_lines"]:
            text += (f"Компилятор проанализировал {plural(len(autopar['loops']), 'цикл', 'цикла', 'циклов')} файла "
                     "`cholesky.cpp` и ")
        else:
            text += "Компилятор "
        text += ("не распараллелил ни одного из них. " if not autopar["parallelized"] and autopar["compiler_lines"]
                 else "не распараллелил ни одного цикла. " if not autopar["parallelized"] else
                 "распараллелил " + and_list(f"цикл в функции `{f}` (строка {line})" for line, f in
                                            autopar["parallelized"]) + ". ")
        if outside and not kernel:
            text += ("Это циклы вне замера времени (заполнение тестовой матрицы, вычисление погрешности); в самом "
                     "разложении не распараллелен ни один цикл. ")
        if autopar["reasons"]:
            text += ("Причины отказа (таблица <<tab:autopar_report>>): "
                     + "; ".join(f"«{reason}» — {count}" for count, reason in autopar["reasons"]) + ". ")
        codes_506 = sorted({f for _, f, code in autopar["loops"] if code == "506"})
        if codes_506:
            text += (f"Код 506 получили циклы в функциях {and_list(f'`{f}`' for f in codes_506)}"
                     + ("; в их теле вызывается `checked_sqrt`, который может выбросить исключение"
                        if set(codes_506) <= {"factor_diagonal_block", "cholesky_point"} else "") + ". ")
    text += ("\n\nКомпилятор распараллеливает цикл, только если сам докажет независимость его итераций. В нашей "
             "программе все блоки хранятся в одном массиве и читаются и пишутся через указатели, внутренние циклы "
             "накапливают суммы, а в телах циклов вызываются функции. То, что блоки разных итераций не пересекаются, "
             "следует из алгоритма, но не из текста программы. Директивы OpenMP варианта А передают компилятору именно "
             "это знание программиста.")
    if not kernel and family == "msvc" and autopar["found"]:
        text += (" Поскольку ни один цикл разложения не распараллелен, программа выполняется последовательно "
                 "и её время от числа потоков не зависит.")
    v["analysis_autopar"] = text

    # --- выводы
    best_mode, best_n, best_pp = max(((mode, n, p) for mode in MODES for n in sizes for p in threads),
                                     key=lambda k: s(*k))
    conclusions = []
    if identical:
        conclusions.append("Распараллеливание не изменило результат: при любом числе потоков и в обеих стратегиях "
                           "фактор $L$ побитно совпадает с последовательным, погрешность — "
                           f"{sci(e1)} при $N = {n1}$.")
    else:
        conclusions.append(f"Погрешность параллельной программы — {sci(e1)} при $N = {n1}$, как у последовательной.")
    conclusions.append(
        f"Вариант А: директивы OpenMP над одним циклом — по блокам столбца — дают относительное ускорение до "
        f"{dec(s(best_mode, best_n, best_pp))} (стратегия `{best_mode}`, $N = {best_n}$, $P = {best_pp}$); ускорение "
        "растёт с размерностью" + (f", а {n_set(sorted({n for _, n, _ in slow}), sizes)} при некоторых $P$ "
                                   "параллельная программа медленнее однопоточной." if slow else "."))
    if p_agree:
        conclusions.append(
            f"До $P = {p_agree}$ ускорение совпадает с моделью распределения блоков: его ограничивают последовательная "
            f"часть шага и неравномерность распределения блоков ($\\alpha$ = {percent(alpha['block', big])} "
            f"для `block` при $N = {big}$).")
    if len(win) > len(sizes) / 2:
        conclusions.append(f"Стратегия `phases` с меньшей последовательной частью быстрее `block` при большинстве $N$"
                           + (f" (на {percent(gain)} при $N = {big}$, $P = {p_cores}$)." if gain >= 0.02 else "."))
    if gap:
        conclusions.append(f"На всех {p_cores} ядрах ускорение ниже модели, а потоки сверх числа ядер его не "
                           "увеличивают: при большом числе потоков ускорение ограничивает уже компьютер, а не "
                           "алгоритм.")
    if s_auto:
        conclusions.append(
            "Вариант Б: автоматическое распараллеливание компилятором "
            + (f"ускорения не дало ($S_{{\\text{{абс}}}}$ = {dec(min(s_auto))}–{dec(max(s_auto))})"
               if max(s_auto) < 1.1 else f"дало ускорение до {dec(max(s_auto))}")
            + (": компилятор не смог доказать независимость итераций ни одного цикла разложения" if not kernel else "")
            + ". Для этого алгоритма распараллеливание требует знания программиста, выраженного директивами.")
    v["analysis_conclusions"] = "\n".join(f"{k}. {c}" for k, c in enumerate(conclusions, 1))

    # --- таблицы
    tables = {}
    autopar_name = "cholesky_autopar" if family == "msvc" else "cholesky_autopar_p<P>"
    autopar_flags = re.sub(r"parallelize-loops=\d+", "parallelize-loops=P", m.flags.get("autopar", "—"))
    builds = [["cholesky_seq", m.flags.get("seq", "—"), "последовательная программа (этап 2): $T_{\\text{посл}}$"],
              ["cholesky_omp", m.flags.get("omp", "—"), "вариант А: директивы OpenMP; $T_1$ и $T_P$"]]
    if "autopar" in m.flags:
        builds.append([autopar_name, autopar_flags, "вариант Б: автоматическое распараллеливание компилятором"])
    tables["builds"] = ("Сборки программы", ["Программа", "Ключи компиляции", "Назначение"], builds,
                        [2700, 3300, 3638], 99)
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
    width = (TABLE_WIDTH - 1000 - 2 * 1500) // len(p_tab)
    for mode in MODES:
        tables[f"speedup_{mode}"] = (
            f"Время и относительное ускорение $S_{{\\text{{отн}}}}$, стратегия {mode} ($b = {b}$)",
            ["$N$", "$T_{\\text{посл}}$, мс", "$T_1$, мс", *(f"$P = {p}$" for p in p_tab)],
            [[str(n), ms(m.t_seq(n)), ms(m.t("omp", mode, n, 1)), *(dec(s(mode, n, p)) for p in p_tab)]
             for n in sizes],
            [1000, 1500, 1500, *([width] * len(p_tab))], 0)
    width = (TABLE_WIDTH - 1000) // (2 * len(fixed))
    tables["speedup_p"] = (
        "Относительное ускорение $S$ и эффективность $E$ от числа потоков, стратегия block",
        ["$P$", *(f"{q}, $N = {n}$" for n in fixed for q in ("$S$", "$E$"))],
        [[str(p), *(x for n in fixed for x in (dec(s("block", n, p)), dec(s("block", n, p) / p)))] for p in threads],
        [1000, *([width] * (2 * len(fixed)))], 0)
    tables["model"] = (
        f"Относительное ускорение и модель распределения блоков при $N = {big}$",
        ["$P$", "block: замер", "block: модель", "phases: замер", "phases: модель"],
        [[str(p), *(x for mode in MODES for x in (dec(s(mode, big, p)), dec(model[mode, big, p])))] for p in within],
        [1238, 2100, 2100, 2100, 2100], 0)
    # подгонка по всем P до числа ядер (как на графике speedup_p) — отдельным столбцом, если диапазон другой
    tables["amdahl"] = (
        "Доля последовательной работы $\\alpha$ (закон Амдала)",
        ["Стратегия", "$N$", *([f"По замерам, $P \\leq {p_cores}$ (как на графике)"] if wide else []),
         f"По замерам, $P \\leq {max(fit)}$", "Модель", "Оценка"],
        [[mode, str(n), *([percent(alpha_cores[mode, n])] if wide else []), percent(alpha[mode, n]),
          percent(alpha_model[mode, n]),
          (f"$1{{,}}5\\, b/N$ = {percent(alpha_theory[mode, n])}" if mode == "block" else
           f"$b^2/N^2$ = {percent(alpha_theory[mode, n])}")] for mode in MODES for n in fixed],
        [1500, 900, 1900, 1700, 1400, 2238] if wide else [1500, 1100, 2400, 2000, 2638], 0)
    if m.autopar_threads:
        width = (TABLE_WIDTH - 1000 - 1800) // len(m.autopar_threads)
        tables["autopar"] = (
            "Вариант Б: абсолютное ускорение программы с автоматическим распараллеливанием",
            ["$N$", "$T_{\\text{посл}}$, мс", *(f"$S_{{\\text{{абс}}}}$, $P = {p}$" for p in m.autopar_threads)],
            [[str(n), ms(m.t_seq(n)), *(dec(m.s_autopar(n, p)) for p in m.autopar_threads)] for n in sizes],
            [1000, 1800, *([width] * len(m.autopar_threads))], 0)
    else:
        tables["autopar"] = ("Вариант Б: абсолютное ускорение", ["$N$", "Результат"],
                             [["—", "замеры не выполнялись"]], [3000, 6638], 0)
    rows = [["распараллелено", str(len(autopar["parallelized"]))]]
    rows += [[f"не распараллелено: {reason}", str(count)] for count, reason in autopar["reasons"]]
    if not autopar["found"]:
        rows = [["отчёт компилятора не найден", "—"]]
    tables["autopar_report"] = ("Отчёт компилятора об автопараллелизации: циклы файла cholesky.cpp",
                                ["Результат", "Число циклов"], rows, [7638, 2000], 1)
    tables["hardware"] = (None, ["Параметр", "Значение"], hardware_rows(info, m), [4819, 4819], 99)
    return v, tables


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


def build(results_dir: Path = RESULTS_DIR, output: Path = OUTPUT):
    m = Measurements(results_dir / "measurements.csv")
    info = read_platform(results_dir / "platform.txt")
    report_file = results_dir / "autopar_report.txt"
    autopar = parse_autopar(report_file.read_text(encoding="utf-8") if report_file.exists() else
                            "Отчёт компилятора об автопараллелизации не найден")
    values, tables = compute(m, info, autopar)
    text = "\n".join(line for line in TEMPLATE_TEXT.read_text(encoding="utf-8").splitlines()
                     if not line.startswith("%"))  # комментарии шаблона
    blocks = parse_blocks(substitute(text, values))

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
