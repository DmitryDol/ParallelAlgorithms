"""
Этап 2: черновик отчёта «Отчёт_ПП_2_Абрамкин_Должиков.docx» из результатов замеров.

Запуск из корня репозитория (после bench.py и plot_speedup.py):
    uv run python lab2_cpp/report/make_report.py

Оформление берётся из отчёта этапа 1: титульный лист (меняется только номер работы), стили, колонтитулы,
вид таблиц, рисунков и листингов. Текст — из report_template.md (его можно править), числа, таблицы и графики —
из lab2_cpp/results/. Формулы LaTeX переводятся в формулы Word программой pandoc (пакет pypandoc-binary).
Разделы «Анализ результатов» собираются из чисел; после нового прогона их нужно перечитать.
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
from plot_speedup import Measurements, ratio  # noqa: E402
from programs import RESULTS_DIR  # noqa: E402

TEMPLATE_DOCX = HERE.parent.parent / "lab1_python" / "report" / "Отчёт_ПП_1_Абрамкин_Должиков.docx"
TEMPLATE_TEXT = HERE / "report_template.md"
OUTPUT = HERE / "Отчёт_ПП_2_Абрамкин_Должиков.docx"
LAB_NUMBER = 2

TABLE_WIDTH = 9638           # ширина таблиц (twips), как в отчёте этапа 1
IMAGE_WIDTH_EMU = 5760000    # ширина рисунков — 16 см
BULLET_NUM_ID = "33"         # маркированный список шаблона
DECIMAL_ABSTRACT_NUM = "0"   # нумерованный список шаблона («1.», «2.»)
VARIANT_PURPOSE = {
    "gcc": {
        "noopt": "«без флага»: оптимизация выключена",
        "opt": "«с флагом»: оптимизация и векторизация под процессор",
        "opt_novec": "то же без автовекторизации — вклад векторизации",
        "opt_fastmath": "то же с разрешением переставлять операции с плавающей точкой",
    },
    "msvc": {
        "noopt": "«без флага»: оптимизация выключена",
        "opt": "«с флагом»: оптимизация по скорости, команды AVX2",
        "opt_novec": "то же без AVX2 (только команды SSE2)",
        "opt_fastmath": "то же с разрешением переставлять операции с плавающей точкой",
    },
}
MACHINE_EPSILON = 2.220446049250313e-16


# ---------------------------------------------------------------- числа
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def dec(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def dec_tex(x, digits=2):
    """Десятичная запятая внутри формулы LaTeX: 0{,}89 (без лишнего пробела)."""
    return dec(x, digits).replace(",", "{,}")


def kb_text(kb):
    return f"{dec(kb / 1024, 0)} МБ" if kb >= 1024 else f"{dec(kb, 0)} КБ"


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


def trend(first, last):
    if last > first * 1.1:
        return "растёт"
    if last < first / 1.1:
        return "уменьшается"
    return "почти не меняется"


def times_phrase(r):
    """1,70 → «быстрее в 1,70 раза»; 0,95 → «медленнее в 1,05 раза»."""
    return f"быстрее в {dec(r)} раза" if r >= 1 else f"медленнее в {dec(1 / r)} раза"


def compiler_text(raw):
    """«MSVC 195136260» (значение _MSC_FULL_VER) → «MSVC 19.51.36260»; остальное без изменений."""
    found = re.fullmatch(r"MSVC (\d{9})", raw.strip())
    if not found:
        return raw
    v = int(found.group(1))
    return f"MSVC {v // 10**7}.{v // 10**5 % 100}.{v % 10**5}"


def power_of_two_dips(sizes, values, threshold=0.95):
    """N — степени двойки, при которых значение (например, производительность) больше чем на 5 % ниже,
    чем у соседних N ряда: признак конфликтных промахов кэша."""
    dips = []
    for k, n in enumerate(sizes):
        neighbours = [values[j] for j in (k - 1, k + 1) if 0 <= j < len(sizes)]
        if n & (n - 1) == 0 and neighbours and values[k] < threshold * min(neighbours):
            dips.append(n)
    return dips


def n_list(ns):
    return ", ".join(map(str, ns))


def n_equals(ns):
    """[1024, 2048] → «$N = 1024$ и $N = 2048$» (перечисление вне формулы — без склеивания через запятую)."""
    parts = [f"$N = {n}$" for n in ns]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " и " + parts[-1]


def trend_text(first, n_first, last, n_last):
    """«растёт с 4,33 при $N = 64$ до 7,08 при $N = 1536$» или «почти не меняется: … и …»."""
    word = trend(first, last)
    if word == "почти не меняется":
        return f"{word}: {dec(first)} при $N = {n_first}$ и {dec(last)} при $N = {n_last}$"
    return f"{word} с {dec(first)} при $N = {n_first}$ до {dec(last)} при $N = {n_last}$"


# ---------------------------------------------------------------- данные
def read_platform(path: Path) -> dict:
    """Строки «ключ: значение» из platform.txt до списка сборок."""
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            break
        key, _, value = line.partition(":")
        info[key.strip()] = value.strip()
    return info


def per_core_kb(text):
    """«8 МБ (4 × 2 МБ)» → 2048; «48 КБ» → 48."""
    found = re.search(r"×\s*([\d,]+)\s*(КБ|МБ)", text or "") or re.match(r"([\d,]+)\s*(КБ|МБ)", text or "")
    if not found:
        return None
    value = float(found.group(1).replace(",", "."))
    return value * 1024 if found.group(2) == "МБ" else value


def hardware_rows(info, m):
    rows = [[key, info[key]] for key in ("Процессор", "Частота процессора", "Физических ядер", "Логических процессоров")
            if key in info]
    rows += [[key, value] for key, value in info.items() if key.startswith("Кэш")]
    rows += [["Операционная система", info.get("ОС", "—")],
             ["Компилятор", compiler_text(m.compiler.get("opt", "—"))],
             ["Язык", "C++ (стандарт C++17)"]]
    return rows


def compute(m: Measurements, info: dict) -> tuple[dict, dict]:
    """Подстановки для шаблона и таблицы."""
    sizes, b, family = m.sizes, m.block, m.family
    n0, n1 = sizes[0], sizes[-1]
    has = set(m.variants)
    fast = m.flags["opt_fastmath"].split()[-1] if "opt_fastmath" in has else ("/fp:fast" if family == "msvc"
                                                                             else "-ffast-math")

    def speedup(num_variant, num_mode, den_variant, den_mode):
        return {n: ratio(m.t(num_variant, num_mode, n), m.t(den_variant, den_mode, n)) for n in sizes}

    def gflops(variant, mode, n):
        return n**3 / 3 / m.t(variant, mode, n) / 1e9

    sa_b = speedup("noopt", "block", "opt", "block")
    sa_p = speedup("noopt", "point", "opt", "point")
    sb_opt = speedup("opt", "point", "opt", "block")
    sb_noopt = speedup("noopt", "point", "noopt", "block")
    perf_block = [gflops("opt", "block", n) for n in sizes]
    dips = power_of_two_dips(sizes, perf_block)
    regular = [n for n in sizes if n not in dips]
    n_ref = regular[-1]  # наибольшая N без провала — для сравнений сборок

    v = {
        "compiler": compiler_text(m.compiler.get("opt", "—")),
        "repeats": str(m.repeats),
        "b": str(b),
        "sizes": n_list(sizes),
    }

    # --- ключи компиляции (разные у MSVC и GCC/Clang)
    if family == "msvc":
        v["flags_explanation"] = (
            "Ключ `/Od` отключает оптимизацию: каждая переменная хранится в памяти, и каждое обращение к ней — "
            "отдельная команда загрузки или записи. Ключ `/O2` включает оптимизацию по скорости: распределение "
            "переменных по регистрам, вынос инвариантов из циклов, встраивание функций и автоматическую векторизацию; "
            "`/arch:AVX2` разрешает команды AVX2 (векторные регистры на 4 числа `double`), без него используются "
            "команды SSE2 (2 числа). Скалярное произведение — это редукция: в режиме `/fp:precise` (по умолчанию) "
            "компилятор сохраняет порядок сложений и поэтому его не векторизует; ключ `/fp:fast` разрешает "
            "переставлять операции с плавающей точкой.")
        v["vector_what"] = f"Вклад набора команд AVX2 и ключа {fast}"
    else:
        v["flags_explanation"] = (
            "Ключ `-O0` отключает оптимизацию: каждая переменная хранится в памяти, и каждое обращение к ней — "
            "отдельная команда загрузки или записи. Ключ `-O3` включает распределение переменных по регистрам, вынос "
            "инвариантов из циклов, встраивание функций и автоматическую векторизацию; `-march=native` разрешает "
            "использовать все расширения набора команд процессора (AVX2, FMA). Скалярное произведение — это редукция, "
            "поэтому без разрешения переставлять сложения (`-ffast-math`) компилятор векторизует только умножения, "
            "а сложения выполняет последовательно в исходном порядке.")
        v["vector_what"] = f"Вклад векторизации и ключа {fast}"

    # --- различия погрешности между сборками
    exact = [var for var in ("noopt", "opt", "opt_novec") if var in has]
    same = all(m.err(var, mode, n) == m.err("noopt", mode, n) for var in exact for mode in ("block", "point")
               for n in sizes)
    if same:
        v["builds_difference"] = (
            f"Сборки без ключа {fast} дают одинаковую погрешность до последнего разряда: компилятор не меняет порядок "
            f"операций. Ключ {fast} меняет порядок суммирования, поэтому погрешность в этой сборке немного отличается.")
    elif family == "gcc":
        v["builds_difference"] = (
            "Небольшие различия между сборками объясняются тем, что с ключом `-march=native` компилятор заменяет пары "
            f"«умножение + сложение» командой FMA с одним округлением, а `{fast}` меняет порядок суммирования.")
    else:
        v["builds_difference"] = ("Небольшие различия между сборками объясняются разным порядком операций "
                                  "с плавающей точкой.")

    # --- выбор размера блока
    sweep = {}
    for (experiment, _, _, n, bb), t in m.time.items():
        if experiment == "block_sweep":
            sweep.setdefault(n, {})[bb] = t
    auto = "подобран" in info.get("Размер блока", "")
    v["block_choice"] = (f"Размер блока $b = {b}$ подобран автоматически: при нём среднее время разложения при "
                         f"$N = {n_list(sorted(sweep))}$ наименьшее (рис. <<fig:time_vs_block.png>>)"
                         if auto and sweep else
                         f"Размер блока задан вручную: $b = {b}$; время при других $b$ показано на рисунке "
                         "<<fig:time_vs_block.png>>")

    # --- корректность
    errors = [m.err(var, mode, n) for var in m.variants for mode in ("block", "point") for n in sizes]
    e0, e1 = m.err("opt", "block", n0), m.err("opt", "block", n1)
    growth = "растёт с увеличением $N$" if e1 > e0 else "почти не зависит от $N$"
    v["analysis_correctness"] = (
        f"Во всём диапазоне $N = {n0} \\ldots {n1}$ погрешность блочного алгоритма в сборке с флагом составляет "
        f"от {sci(e0)} до {sci(e1)} и, как и на этапе 1, {growth} из-за накопления ошибок округления "
        f"в скалярных произведениях. Наибольшая погрешность среди всех сборок и режимов — {sci(max(errors))}, "
        f"то есть не более {round(max(errors) / MACHINE_EPSILON)} машинных эпсилон, тогда как ошибка "
        f"в программе дала бы погрешность порядка единицы. Ускорение во всех случаях получено без потери точности.")

    # --- провалы при N — степенях двойки
    dips_text = ""
    if dips:
        worst = min(dips, key=lambda n: perf_block[sizes.index(n)] / perf_block[sizes.index(n) - 1])
        k = sizes.index(worst)
        point_drop = gflops("opt", "point", worst) / gflops("opt", "point", sizes[k - 1])
        dips_text = (
            f"При {n_equals(dips)} (степени двойки) производительность блочного алгоритма в сборке с флагом "
            f"заметно ниже, чем при соседних $N$: например, {dec(perf_block[k])} GFLOP/s при $N = {worst}$ против "
            f"{dec(perf_block[k - 1])} при $N = {sizes[k - 1]}$. Строка матрицы занимает $8N$ байт; при таких $N$ это "
            "число кратно 4 КБ, и элементы одного столбца в соседних строках попадают в одно и то же множество "
            "ассоциативного кэша L1 — строки блока вытесняют друг друга (конфликтные промахи). ")
        dips_text += ("Точечный алгоритм читает строки последовательно и от этого почти не страдает."
                      if point_drop > 0.95 else
                      f"Точечный алгоритм страдает меньше: его производительность снижается в "
                      f"{dec(1 / point_drop)} раза.")

    # --- вариант А
    best_n = max(regular, key=lambda n: sa_b[n])
    lo_a, hi_a = min(sa_b[n] for n in regular), max(sa_b[n] for n in regular)
    text = (f"Для блочного алгоритма ускорение от флага компиляции {trend_text(sa_b[n0], n0, sa_b[n_ref], n_ref)} "
            f"(наибольшее значение {dec(sa_b[best_n])} при $N = {best_n}$). ")
    if sa_b[n_ref] > sa_b[n0] * 1.1 and sa_b[n_ref] >= 0.9 * sa_b[best_n]:
        text += ("При малых $N$ в оптимизированной программе заметна доля коротких циклов и неполных блоков, "
                 "а при больших $N$ основное время занимает умножение блоков, данные которых находятся в кэш-памяти, "
                 "поэтому ускорение выходит на плато. ")
    if dips:
        text += (f"При {n_equals(dips)} ускорение меньше ({'; '.join(dec(sa_b[n]) for n in dips)}) из-за "
                 "конфликтных промахов кэша в оптимизированной программе (п. «Ускорение блочного алгоритма»). ")
    noopt_gflops = [gflops("noopt", "block", n) for n in sizes]
    text += (f"Программа без флага ограничена числом выполняемых команд: её производительность "
             f"({dec(min(noopt_gflops))}–{dec(max(noopt_gflops))} GFLOP/s) почти не зависит от $N$.\n\n")
    if sa_p[n1] < sa_p[n0] / 1.1:
        text += (f"Для точечного алгоритма ускорение, напротив, уменьшается с {dec(sa_p[n0])} до {dec(sa_p[n1])}: "
                 "при больших $N$ строки матрицы перестают помещаться в кэш-память, и оптимизированная программа "
                 "значительную часть времени ожидает данные из оперативной памяти — ключи компиляции не ускоряют "
                 "обращения к памяти.")
    else:
        text += f"Для точечного алгоритма ускорение {trend_text(sa_p[n0], n0, sa_p[n1], n1)}."
    v["analysis_flags"] = text

    # --- вариант Б
    best_b_n = max(sizes, key=lambda n: sb_opt[n])
    text = (f"В сборке с флагом ускорение блочного алгоритма относительно точечного "
            f"{trend_text(sb_opt[n0], n0, sb_opt[best_b_n], best_b_n)}. ")
    cross = next((n for n in regular if sb_opt[n] >= 1.0), None)
    if cross and cross != n0:
        slower = [n for n in sizes if n > cross and sb_opt[n] < 1.0]
        text += (f"При $N < {cross}$ блочный алгоритм медленнее точечного: матрица целиком помещается в кэш, "
                 f"и блочность только добавляет накладные расходы (скалярные произведения длины $b = {b}$ вместо "
                 f"длины до $N$). Начиная с $N = {cross}$ блочный алгоритм быстрее"
                 + (f", кроме {n_equals(slower)}. " if slower else ". "))
    elif not cross:
        text += "Во всём диапазоне блочный алгоритм не быстрее точечного. "
    l2 = per_core_kb(info.get("Кэш L2"))
    if l2 and cross:
        n_l2 = int(math.sqrt(l2 * 1024 / 4))
        text += (f"Выигрыш растёт по мере того, как данные перестают помещаться в кэш-память: нижний треугольник "
                 f"матрицы (около $4N^2$ байт) не помещается в кэш L2 ({kb_text(l2)} на ядро) начиная с "
                 f"$N \\approx {n_l2}$. Точечный алгоритм при этом всё больше ожидает данные из оперативной памяти, "
                 f"а блочный продолжает работать с блоками $b \\times b$, находящимися в кэше. ")
    if dips_text:
        text += "\n\n" + dips_text
    if "opt_fastmath" in has:
        sb_fast = speedup("opt_fastmath", "point", "opt_fastmath", "block")
        if max(sb_fast.values()) < 1.0:
            text += (f"\n\nС ключом {fast} блочный алгоритм медленнее точечного при всех $N$ "
                     f"($S_Б = {dec_tex(min(sb_fast.values()))} \\ldots {dec_tex(max(sb_fast.values()))}$): "
                     f"векторизованное скалярное произведение тем эффективнее, чем длиннее векторы, а у блочного "
                     f"алгоритма их длина равна $b = {b}$, у точечного — до $N$.")
    lo, hi = min(sb_noopt.values()), max(sb_noopt.values())
    text += (f"\n\nВ сборке без флага $S_Б = {dec_tex(lo)} \\ldots {dec_tex(hi)}$: неоптимизированная программа "
             "ограничена числом выполняемых команд, а не обменом с памятью, поэтому уменьшение числа обращений "
             "к памяти за счёт блочности не даёт выигрыша. Таким образом, эффект блочного алгоритма проявляется "
             "только вместе с оптимизацией.")
    v["analysis_block"] = text

    # --- векторизация и размер блока
    text = ""
    if "opt_novec" in has:
        r_b = ratio(m.t("opt_novec", "block", n_ref), m.t("opt", "block", n_ref))
        r_p = ratio(m.t("opt_novec", "point", n_ref), m.t("opt", "point", n_ref))
        if family == "msvc":
            text += (f"Сборка с `/arch:AVX2` по сравнению со сборкой без него ($N = {n_ref}$): блочный алгоритм "
                     f"{times_phrase(r_b)}, точечный — {times_phrase(r_p)}. ")
            if max(r_b, r_p, 1 / r_b, 1 / r_p) < 1.1:
                text += ("Без ключа `/fp:fast` скалярное произведение не векторизуется, поэтому более широкие "
                         "векторные регистры AVX2 почти не влияют на время. ")
        else:
            text += (f"Векторизация (сравнение со сборкой без неё, $N = {n_ref}$): блочный алгоритм "
                     f"{times_phrase(r_b)}, точечный — {times_phrase(r_p)}. Без `{fast}` векторизуются только "
                     "умножения, а сложения остаются последовательными. ")
    if "opt_fastmath" in has:
        f_b = ratio(m.t("opt", "block", n_ref), m.t("opt_fastmath", "block", n_ref))
        f_p = ratio(m.t("opt", "point", n_ref), m.t("opt_fastmath", "point", n_ref))
        text += (f"С ключом `{fast}`, разрешающим переставлять сложения, блочный алгоритм {times_phrase(f_b)}, "
                 f"точечный — {times_phrase(f_p)}: только с ним скалярное произведение векторизуется полностью "
                 "(частичные суммы в векторном регистре), при этом результат меняется в последних разрядах. ")
    if sweep and all(b in times for times in sweep.values()):
        best = {n: min(times, key=times.get) for n, times in sorted(sweep.items())}
        excess = max(sweep[n][b] / sweep[n][best[n]] - 1 for n in sweep)
        if len(set(best.values())) == 1:
            text += (f"Наименьшее время разложения при всех $N$ подбора получено при $b = {next(iter(best.values()))}$"
                     + (" — он и используется. " if excess == 0 else ". "))
        else:
            text += ("Наименьшее время разложения при подборе получено при "
                     + ", ".join(f"$b = {bb}$ ($N = {n}$)" for n, bb in best.items()) + ". ")
        if excess == 0:
            pass
        elif excess <= 0.05:
            text += (f"Минимум пологий: при $b = {b}$ время больше наименьшего не более чем на "
                     f"{dec(100 * excess, 1)} %. ")
        elif auto:
            text += (f"Выбран $b = {b}$ — с наименьшим средним временем по всем $N$ подбора (при отдельных $N$ "
                     f"проигрыш до {dec(100 * excess, 1)} %). ")
        else:
            text += (f"Выбранный $b = {b}$ на этом компьютере не оптимален: время больше наименьшего на "
                     f"{dec(100 * excess, 1)} %. ")
    l1 = per_core_kb(info.get("Кэш L1 данных") or info.get("Кэш L1 (данные и команды)"))
    kb = 3 * 8 * b * b / 1024
    text += f"Три блока ${b} \\times {b}$, участвующие в модификации, занимают {dec(kb, 0)} КБ"
    text += f" и помещаются в кэш L1 ({dec(l1, 0)} КБ на ядро)." if l1 and kb <= l1 else "."
    v["analysis_vector"] = text

    # --- выводы
    lo_p, hi_p = min(sa_p.values()), max(sa_p.values())
    conclusions = [
        f"Программа на C++ воспроизводит результаты этапа 1: фактор $L$ совпадает с эталонным с точностью "
        f"$10^{{-13}}$, погрешность того же порядка ({sci(e1)} при $N = {n1}$).",
        f"Флаг компиляции ускоряет блочный алгоритм в {dec(lo_a)}–{dec(hi_a)} раза"
        + (f" (при {n_equals(dips)} меньше из-за конфликтных промахов кэша)" if dips else "")
        + f", точечный — в {dec(lo_p)}–{dec(hi_p)} раза.",
        (f"Блочный алгоритм в сборке с флагом быстрее точечного до {dec(sb_opt[best_b_n])} раза ($N = {best_b_n}$); "
         "выигрыш появляется, когда матрица перестаёт помещаться в кэш-память, и отсутствует в сборке без флага."
         if sb_opt[best_b_n] >= 1.05 else
         "Блочный алгоритм в сборке с флагом не даёт заметного выигрыша относительно точечного."),
        f"Наибольшая производительность блочного алгоритма — {dec(max(perf_block), 1)} GFLOP/s в сборке с флагом"
        + (f" и {dec(max(gflops('opt_fastmath', 'block', n) for n in sizes), 1)} GFLOP/s с `{fast}`."
           if "opt_fastmath" in has else "."),
    ]
    if dips:
        conclusions.append("Размерности, равные степеням двойки, неудачны для блочного алгоритма из-за конфликтных "
                           "промахов кэша; на практике строки дополняют несколькими элементами («набивка»).")
    v["analysis_conclusions"] = "\n".join(f"{k}. {c}" for k, c in enumerate(conclusions, 1))

    # --- таблицы
    purpose = VARIANT_PURPOSE[family]
    tables = {}
    tables["builds"] = (
        "Сборки программы", ["Программа", "Ключи компиляции", "Назначение"],
        [[f"cholesky_{var}", m.flags[var], purpose[var]] for var in m.variants],
        [3100, 3400, 3138], 99)
    tables["errors"] = (
        f"Погрешность блочного алгоритма ($b = {b}$) в разных сборках",
        ["$N$", *(m.name(var) for var in m.variants)],
        [[str(n), *(sci(m.err(var, "block", n)) for var in m.variants)] for n in sizes],
        [1200, *([(TABLE_WIDTH - 1200) // len(m.variants)] * len(m.variants))], 0)
    tables["flags"] = (
        f"Время разложения и ускорение от флага компиляции ($b = {b}$)",
        ["$N$", "$T$ без флага, мс", "$T$ с флагом, мс", "$S_А$, блочный", "$S_А$, точечный"],
        [[str(n), ms(m.t("noopt", "block", n)), ms(m.t("opt", "block", n)), dec(sa_b[n]), dec(sa_p[n])]
         for n in sizes],
        [1200, 2200, 2200, 2019, 2019], 0)
    tables["block"] = (
        f"Время точечного и блочного ($b = {b}$) алгоритмов в сборке с флагом и ускорение блочного алгоритма",
        ["$N$", "$T$ точечный, мс", "$T$ блочный, мс", "$S_Б$ с флагом", "$S_Б$ без флага"],
        [[str(n), ms(m.t("opt", "point", n)), ms(m.t("opt", "block", n)), dec(sb_opt[n]), dec(sb_noopt[n])]
         for n in sizes],
        [1200, 2200, 2200, 2019, 2019], 0)
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


def latex_to_omml(snippets: list) -> list:
    """[(tex, display)] → [OMML] одним вызовом pandoc."""
    if not snippets:
        return []
    md = "\n\n".join(f"$${tex}$$" if display else f"${tex}$" for tex, display in snippets)
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
    values, tables = compute(m, info)
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
