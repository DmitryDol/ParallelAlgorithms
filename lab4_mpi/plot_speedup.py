"""
Этап 4: таблицы и графики ускорения по results/measurements.csv (его пишет bench.py).

Запуск из корня репозитория:
    uv run python lab4_mpi/plot_speedup.py

Ускорение считается по минимальному времени из повторов:
  относительное S_отн(P) = T_1 / T_P — та же программа MPI на одном процессе;
  абсолютное    S_абс(P) = T_посл / T_P — последовательная программа (алгоритм этапа 2, те же ключи).
Результаты (в results/):
  speedup_n, speedup_n_abs — стратегия block: S_отн(N) и S_абс(N) для каждого P;
  speedup_p, speedup_p_abs — S(P) при нескольких N, идеальное S = P и закон Амдала;
  efficiency_p             — эффективность E = S_отн / P и модель распределения строк (cyclic_model_speedup);
  strategies               — стратегии block и phases при наибольшем P на физических ядрах;
  mpi_vs_omp               — MPI против OpenMP (этап 3), если этап 3 замерен на том же процессоре;
  scheme                   — схема: циклическое распределение блочных строк и рассылка строки j (для отчёта);
  speedup_block.csv, speedup_phases.csv, speedup_p.csv, amdahl.csv, model.csv, mpi_vs_omp.csv — таблицы.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter

from programs import HERE, RESULTS_DIR

STRATEGY_NAMES = {"block": "рассылка строки j после разложения L_jj",
                  "phases": "сначала внедиагональная часть строки j, затем L_jj"}
STRATEGY_COLORS = {"block": "#1f77b4", "phases": "#ff7f0e"}
FIXED_N = [1000, 2000, 3000]
OPENMP_RESULTS = HERE.parent / "lab3_openmp" / "results"


def read_platform(path: Path) -> dict:
    """Строки «ключ: значение» из platform.txt до списка сборок."""
    info = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                break
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    return info


def ratio(numerator, denominator):
    return numerator / denominator if numerator and denominator else None


def amdahl_alpha(threads, speedups):
    """Доля последовательной работы α по закону Амдала 1/S = α + (1 − α)/P (метод наименьших квадратов
    для y − x = α(1 − x), где x = 1/P, y = 1/S); точки с P = 1 не несут информации."""
    points = [(1 / p, 1 / s) for p, s in zip(threads, speedups) if p > 1 and s]
    denominator = sum((1 - x) ** 2 for x, _ in points)
    return sum((1 - x) * (y - x) for x, y in points) / denominator if denominator else None


def amdahl_speedup(alpha, p):
    return 1 / (alpha + (1 - alpha) / p)


def cyclic_model_speedup(n, b, procs, strategy):
    """Модель ускорения, которое допускает распределение работы (без затрат на обмены): время операции над блоком
    пропорционально числу её операций, процессы равноскоростные. Блочная строка i — у процесса i mod P; на шаге j
    каждый процесс считает свои строки i > j, шаг длится столько, сколько у самого нагруженного процесса;
    block: Ã_jj и L_jj владелец считает, пока остальные ждут; phases: Ã_jj — вместе с остальными модификациями,
    последовательно только L_jj."""
    blocks = -(-n // b)
    t1 = tp = 0.0
    for jb in range(blocks):
        mj = min(b, n - jb * b)

        def modify(ib, diagonal=False):  # скалярные произведения длины jb·b (и копирование A_ij)
            mi = min(b, n - ib * b)
            pairs = mi * (mi + 1) / 2 if diagonal else mi * mj
            return pairs * (2 * jb * b + 1)

        def solve(ib):
            return min(b, n - ib * b) * mj * mj

        factor, root = mj ** 3 / 3, jb % procs
        mods, solves = [0.0] * procs, [0.0] * procs
        for ib in range(jb + 1, blocks):
            mods[ib % procs] += modify(ib)
            solves[ib % procs] += solve(ib)
        t1 += modify(jb, True) + factor + sum(mods) + sum(solves)
        if strategy == "block":
            tp += modify(jb, True) + factor + max(m + s for m, s in zip(mods, solves))
        else:
            mods[root] += modify(jb, True)
            tp += max(mods) + factor + max(solves)
    return t1 / tp


class Measurements:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.time, self.median, self.error, self.sweep = {}, {}, {}, {}
        self.flags, self.compiler, self.mpi = {}, {}, ""
        for r in rows:
            n, p = int(r["N"]), int(r["procs"])
            if r["experiment"] == "block_sweep":
                self.sweep.setdefault(n, {})[int(r["b"])] = float(r["t_min"])
                continue
            key = (r["experiment"], r["mode"], n, p)
            self.time[key] = float(r["t_min"])
            self.median[key] = float(r["t_median"])
            self.error[key] = float(r["err_L"])
            self.flags[r["experiment"]] = r["flags"]
            self.compiler[r["experiment"]] = r["compiler"]
            if r["experiment"] == "mpi":
                self.mpi = r["mpi"]
                self.block = int(r["b"])
        self.repeats = min(int(r["repeats"]) for r in rows)
        self.sizes = sorted({k[2] for k in self.time if k[0] == "mpi"})
        self.procs = sorted({k[3] for k in self.time if k[0] == "mpi"})

    def t(self, experiment, mode, n, p):
        return self.time.get((experiment, mode, n, p))

    def t_seq(self, n):
        return self.t("seq", "block", n, 1)

    def s_rel(self, mode, n, p):
        return ratio(self.t("mpi", mode, n, 1), self.t("mpi", mode, n, p))

    def s_abs(self, mode, n, p):
        return ratio(self.t_seq(n), self.t("mpi", mode, n, p))

    def fixed_sizes(self):
        chosen = [n for n in FIXED_N if n in self.sizes]
        return chosen or self.sizes[-3:]


def openmp_speedups(results_dir: Path, processor: str) -> dict:
    """S_отн этапа 3 {(режим, N, P): S}, если этап 3 замерен на том же процессоре; иначе пусто."""
    path = results_dir / "measurements.csv"
    if not path.exists() or read_platform(results_dir / "platform.txt").get("Процессор") != processor:
        return {}
    with open(path, encoding="utf-8") as f:
        times = {(r["mode"], int(r["N"]), int(r["threads"])): float(r["t_min"])
                 for r in csv.DictReader(f) if r["experiment"] == "omp"}
    return {key: ratio(times.get((key[0], key[1], 1)), t) for key, t in times.items()}


def new_axes(title, ylabel, xlabel="Размерность матрицы N"):
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", linestyle="-", alpha=0.35)
    return fig, ax


def log2_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values], fontsize=11)
    ax.xaxis.set_minor_formatter(NullFormatter())


def procs_ticks(ax, values):
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values], fontsize=11)


def save(fig, ax, out_dir, name):
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(x, digits=3):
    return "" if x is None else f"{x:.{digits}f}"


def physical_cores(info):
    try:
        return int(info.get("Физических ядер", ""))
    except ValueError:
        return None


def scheme_figure(blocks=8, procs=3, step=3):
    """Схема для отчёта: блочная строка i — у процесса i mod P; на шаге j владелец рассылает строку j (L_j0 … L_jj),
    каждый процесс считает свои блоки L_ij, i > j. Вычисленные раньше блоки — светлее, ещё не вычисленные — белые.
    Возвращает (рисунок, оси, легенду)."""
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    for i in range(blocks):
        owner = i % procs
        for k in range(i + 1):
            done, current = k < step, k == step and i >= step
            face = colors[owner] if done or current else "white"
            alpha = 0.35 if done else 0.85 if current else 1.0
            ax.add_patch(Rectangle((k, i), 1, 1, facecolor=face, alpha=alpha, edgecolor="gray", linewidth=1))
            if current and i > step:
                ax.add_patch(Rectangle((k, i), 1, 1, fill=False, hatch="//", edgecolor="black", linewidth=0))
            ax.text(k + 0.5, i + 0.5, f"$L_{{{i}{k}}}$", ha="center", va="center", fontsize=10,
                    color="black" if done or current else "gray")
        ax.text(-0.25, i + 0.5, f"строка {i} — процесс {owner}", ha="right", va="center", fontsize=11,
                color=colors[owner])
    ax.add_patch(Rectangle((0, step), step + 1, 1, fill=False, edgecolor="black", linewidth=3.5))
    ax.annotate(f"MPI_Bcast: процесс {step % procs} рассылает строку j = {step}\n(L_{step}0 … L_{step}{step}) всем процессам;\n"
                f"затем каждый процесс считает свои\nблоки L_i{step}, i > {step} (заштрихованы)",
                xy=(step + 1, step + 0.5), xytext=(step + 1.6, step - 1.4), fontsize=12, va="center",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
    ax.set_xlim(-4.2, blocks + 5.2)
    ax.set_ylim(blocks + 0.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Циклическое распределение блочных строк: P = {procs}, p = {blocks}, шаг j = {step}",
                 fontsize=15, pad=12)
    legend = [Patch(facecolor=colors[r], alpha=0.85, label=f"процесс {r}") for r in range(procs)]
    legend += [Patch(facecolor="white", edgecolor="black", hatch="//", label="блоки L_ij, i > j: считаются на шаге j"),
               Patch(facecolor="white", edgecolor="gray", label="ещё не вычислены")]
    # левый нижний угол легенды — справа от последнего блока: не наезжает на блоки при любой ширине шрифта
    legend = ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(blocks + 0.3, blocks),
                       bbox_transform=ax.transData, fontsize=11)
    return fig, ax, legend


def draw_scheme(out_dir: Path, **kwargs):
    fig, _, _ = scheme_figure(**kwargs)
    fig.tight_layout()
    fig.savefig(out_dir / "scheme.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Графики ускорения этапа 4.")
    parser.add_argument("-d", "--results-dir", type=Path, default=RESULTS_DIR, help="каталог результатов")
    parser.add_argument("--openmp-dir", type=Path, default=OPENMP_RESULTS, help="результаты этапа 3 для сравнения")
    args = parser.parse_args(argv)
    out = args.results_dir
    m = Measurements(out / "measurements.csv")
    info = read_platform(out / "platform.txt")
    cores = physical_cores(info)
    parallel = [p for p in m.procs if p > 1]
    within = [p for p in m.procs if cores is None or p <= cores]
    colors = plt.cm.tab10.colors

    # --- S(N) для каждого P
    for name, func, label in (("speedup_n", m.s_rel, "относительное"), ("speedup_n_abs", m.s_abs, "абсолютное")):
        fig, ax = new_axes(f"MPI, стратегия block: {label} ускорение, b = {m.block}", "Ускорение S")
        for p, color in zip(parallel, colors):
            points = [(n, func("block", n, p)) for n in m.sizes if func("block", n, p)]
            beyond = cores is not None and p > cores  # процессов больше, чем ядер: делят ядра (Hyper-Threading)
            ax.plot(*zip(*points), "s--" if beyond else "o-", color=color, linewidth=2.2, markersize=7,
                    label=f"P = {p}" + (" (больше, чем ядер)" if beyond else ""))
        ax.axhline(1.0, color="gray", linewidth=1)
        ax.set_ylim(bottom=0)
        log2_ticks(ax, m.sizes)
        save(fig, ax, out, name)

    for mode in ("block", "phases"):
        header = ["N", "T_посл, с", "T_1, с", *(f"S_отн P={p}" for p in parallel), *(f"S_абс P={p}" for p in parallel)]
        rows = [[n, fmt(m.t_seq(n), 6), fmt(m.t("mpi", mode, n, 1), 6),
                 *(fmt(m.s_rel(mode, n, p)) for p in parallel), *(fmt(m.s_abs(mode, n, p)) for p in parallel)]
                for n in m.sizes]
        write_csv(out / f"speedup_{mode}.csv", header, rows)

    # --- S(P) при фиксированных N, идеальное ускорение и закон Амдала
    fixed = m.fixed_sizes()
    amdahl_rows = []
    for name, func, label in (("speedup_p", m.s_rel, "относительное"), ("speedup_p_abs", m.s_abs, "абсолютное")):
        fig, ax = new_axes(f"MPI, стратегия block: {label} ускорение от числа процессов, b = {m.block}",
                           "Ускорение S", "Число процессов P")
        ax.plot(m.procs, m.procs, ":", color="black", label="идеальное S = P")
        for n, color in zip(fixed, ("#1f77b4", "#ff7f0e", "#2ca02c")):
            ax.plot(m.procs, [func("block", n, p) for p in m.procs], "o-", color=color, linewidth=2.2, markersize=7,
                    label=f"N = {n}")
            if name == "speedup_p":  # закон Амдала — для относительного ускорения
                alpha = amdahl_alpha(within, [func("block", n, p) for p in within])
                if alpha is not None and alpha > 0:
                    ax.plot(m.procs, [amdahl_speedup(alpha, p) for p in m.procs], "--", color=color, linewidth=1.2,
                            label=f"закон Амдала, α = {100 * alpha:.1f} %")
        if cores and cores < max(m.procs):
            ax.axvline(cores, color="gray", linestyle="--", linewidth=1.2, label=f"физических ядер: {cores}")
        ax.set_ylim(bottom=0)
        procs_ticks(ax, m.procs)
        save(fig, ax, out, name)

    for mode in ("block", "phases"):
        for n in fixed:
            alpha = amdahl_alpha(within, [m.s_rel(mode, n, p) for p in within])
            amdahl_rows.append([mode, n, fmt(alpha, 5), fmt(1 / alpha if alpha and alpha > 0 else None, 1)])
    write_csv(out / "amdahl.csv", ["стратегия", "N", "alpha", "S_max = 1/alpha"], amdahl_rows)

    header = ["P", *(f"{kind} N={n}" for n in fixed for kind in ("S_отн", "S_абс", "E_отн"))]
    rows = [[p, *(v for n in fixed for v in (fmt(m.s_rel("block", n, p)), fmt(m.s_abs("block", n, p)),
                                               fmt(ratio(m.s_rel("block", n, p), p))))] for p in m.procs]
    write_csv(out / "speedup_p.csv", header, rows)

    # --- эффективность и модель распределения строк
    fig, ax = new_axes(f"MPI, стратегия block: эффективность E = S_отн / P, b = {m.block}", "Эффективность E",
                       "Число процессов P")
    for n, color in zip(fixed, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        ax.plot(m.procs, [ratio(m.s_rel("block", n, p), p) for p in m.procs], "o-", color=color, linewidth=2.2,
                markersize=7, label=f"N = {n}")
        # модель — только до числа ядер: она считает процессы равноскоростными, а на одном ядре два процесса — нет
        ax.plot(within, [cyclic_model_speedup(n, m.block, p, "block") / p for p in within], "--", color=color,
                linewidth=1.2, label=f"модель распределения строк, N = {n}")
    if cores and cores < max(m.procs):
        ax.axvline(cores, color="gray", linestyle="--", linewidth=1.2, label=f"физических ядер: {cores}")
    ax.set_ylim(0, 1.1)
    procs_ticks(ax, m.procs)
    save(fig, ax, out, "efficiency_p")
    model_rows = []
    for mode in ("block", "phases"):
        for n in fixed:
            for p in within:
                model = cyclic_model_speedup(n, m.block, p, mode)
                model_rows.append([mode, n, p, fmt(model), fmt(m.s_rel(mode, n, p)),
                                   fmt(ratio(m.s_rel(mode, n, p), model))])
    write_csv(out / "model.csv", ["стратегия", "N", "P", "S_модель", "S_отн", "S_отн / S_модель"], model_rows)

    # --- стратегии
    p_cmp = max(within)
    fig, ax = new_axes(f"Сравнение стратегий MPI: относительное ускорение при P = {p_cmp}, b = {m.block}",
                       "Ускорение S")
    for mode in ("block", "phases"):
        points = [(n, m.s_rel(mode, n, p_cmp)) for n in m.sizes if m.s_rel(mode, n, p_cmp)]
        ax.plot(*zip(*points), "o-", color=STRATEGY_COLORS[mode], linewidth=2.2, markersize=7,
                label=f"{mode}: {STRATEGY_NAMES[mode]}")
    ax.axhline(1.0, color="gray", linewidth=1)
    ax.set_ylim(bottom=0)
    log2_ticks(ax, m.sizes)
    save(fig, ax, out, "strategies")

    # --- MPI против OpenMP (этап 3 на том же процессоре)
    omp = openmp_speedups(args.openmp_dir, info.get("Процессор", ""))
    big = fixed[-1]
    common = [p for p in m.procs if all(omp.get((mode, big, p)) for mode in ("block", "phases"))]
    if omp and len(common) > 1:
        fig, ax = new_axes(f"MPI и OpenMP (этап 3): относительное ускорение при N = {big}", "Ускорение S",
                           "Число процессов (MPI) или потоков (OpenMP) P")
        ax.plot(common, common, ":", color="black", label="идеальное S = P")
        for mode in ("block", "phases"):
            ax.plot(common, [m.s_rel(mode, big, p) for p in common], "o-", color=STRATEGY_COLORS[mode],
                    linewidth=2.2, markersize=7, label=f"MPI, {mode}")
            ax.plot(common, [omp[mode, big, p] for p in common], "s--", color=STRATEGY_COLORS[mode], linewidth=1.8,
                    markersize=6, label=f"OpenMP, {mode}")
        if cores and cores < max(common):
            ax.axvline(cores, color="gray", linestyle="--", linewidth=1.2, label=f"физических ядер: {cores}")
        ax.set_ylim(bottom=0)
        procs_ticks(ax, common)
        save(fig, ax, out, "mpi_vs_omp")
        write_csv(out / "mpi_vs_omp.csv", ["P", "MPI block", "OpenMP block", "MPI phases", "OpenMP phases"],
                  [[p, fmt(m.s_rel("block", big, p)), fmt(omp["block", big, p]), fmt(m.s_rel("phases", big, p)),
                    fmt(omp["phases", big, p])] for p in common])
    else:
        print("Сравнение с OpenMP пропущено: нет результатов этапа 3 на этом же процессоре")

    draw_scheme(out)
    print(f"Графики и таблицы сохранены в {out}")
    for n in fixed:
        best = max(m.procs, key=lambda p: m.s_rel("block", n, p) or 0)
        print(f"  N = {n}: наибольшее S_отн = {m.s_rel('block', n, best):.2f} при P = {best}, "
              f"S_абс = {m.s_abs('block', n, best):.2f}")


if __name__ == "__main__":
    main()
