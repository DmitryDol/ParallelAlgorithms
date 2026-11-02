"""
Этап 3: таблицы и графики ускорения по results/measurements.csv (его пишет bench.py).

Запуск из корня репозитория:
    uv run python lab3_openmp/plot_speedup.py

Ускорение считается по минимальному времени из повторов:
  относительное S_отн(P) = T_1 / T_P — та же программа omp на одном потоке;
  абсолютное    S_абс(P) = T_посл / T_P — последовательная сборка seq (лучшая последовательная программа).
Результаты (в results/):
  speedup_n, speedup_n_abs — вариант А, стратегия block: S_отн(N) и S_абс(N) для каждого P;
  speedup_p, speedup_p_abs — S(P) при нескольких N, идеальное S = P и закон Амдала;
  efficiency_p             — эффективность E = S_отн / P и модель распределения блоков (static_schedule_speedup);
  strategies               — стратегии block и phases при наибольшем P на физических ядрах;
  autopar                  — вариант Б: S_абс(N) сборок с автопараллелизацией;
  speedup_block.csv, speedup_phases.csv, speedup_p.csv, amdahl.csv, model.csv, autopar.csv — таблицы.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from programs import RESULTS_DIR

STRATEGY_NAMES = {"block": "поддиагональные блоки параллельно", "phases": "модификации и решения параллельно"}
STRATEGY_COLORS = {"block": "#1f77b4", "phases": "#ff7f0e"}
FIXED_N = [1000, 2000, 3000]


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


def static_schedule_speedup(n, b, p, strategy):
    """Модель ускорения, которое допускает сам алгоритм: время операции над блоком пропорционально числу её
    операций, потоки равноскоростные, накладных расходов нет. На шаге j каждый параллельный цикл ждёт поток
    с наибольшей порцией: schedule(static) отдаёт потоку до ⌈m/P⌉ из m соседних итераций."""
    blocks = -(-n // b)

    def chunks_time(costs):  # время параллельного цикла при schedule(static)
        q = -(-len(costs) // p)
        return max(sum(costs[k:k + q]) for k in range(0, len(costs), q)) if costs else 0.0

    t1 = tp = 0.0
    for jb in range(blocks):
        mj = min(b, n - jb * b)

        def modify(ib, diagonal=False):  # скалярные произведения длины jb·b (и копирование A_ij)
            mi = min(b, n - ib * b)
            pairs = mi * (mi + 1) / 2 if diagonal else mi * mj
            return pairs * (2 * jb * b + 1)

        def solve(ib):
            return min(b, n - ib * b) * mj * mj

        factor = mj ** 3 / 3
        below = range(jb + 1, blocks)
        if strategy == "block":
            costs = [modify(ib) + solve(ib) for ib in below]
            t1 += modify(jb, True) + factor + sum(costs)
            tp += modify(jb, True) + factor + chunks_time(costs)
        else:  # phases: модификации столбца → факторизация L_jj → треугольные решения
            mods, solves = [modify(jb, True)] + [modify(ib) for ib in below], [solve(ib) for ib in below]
            t1 += sum(mods) + factor + sum(solves)
            tp += chunks_time(mods) + factor + chunks_time(solves)
    return t1 / tp


class Measurements:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.time, self.error, self.sweep = {}, {}, {}
        self.flags, self.compiler, self.openmp = {}, {}, ""
        for r in rows:
            n, p = int(r["N"]), int(r["threads"])
            if r["experiment"] == "block_sweep":
                self.sweep.setdefault(n, {})[int(r["b"])] = float(r["t_min"])
                continue
            key = (r["experiment"], r["mode"], n, p)  # сборка однозначно задана экспериментом и P
            self.time[key] = float(r["t_min"])
            self.error[key] = float(r["err_L"])
            self.flags[r["experiment"]] = r["flags"]
            self.compiler[r["experiment"]] = r["compiler"]
            if r["experiment"] == "omp":
                self.openmp = r["openmp"]
                self.block = int(r["b"])
        self.repeats = min(int(r["repeats"]) for r in rows)
        self.sizes = sorted({k[2] for k in self.time if k[0] == "omp"})
        self.threads = sorted({k[3] for k in self.time if k[0] == "omp"})
        self.autopar_threads = sorted({k[3] for k in self.time if k[0] == "autopar"})

    def t(self, experiment, mode, n, p):
        return self.time.get((experiment, mode, n, p))

    def t_seq(self, n):
        return self.t("seq", "block", n, 1)

    def s_rel(self, mode, n, p):
        return ratio(self.t("omp", mode, n, 1), self.t("omp", mode, n, p))

    def s_abs(self, mode, n, p):
        return ratio(self.t_seq(n), self.t("omp", mode, n, p))

    def s_autopar(self, n, p):
        return ratio(self.t_seq(n), self.t("autopar", "block", n, p))

    def fixed_sizes(self):
        chosen = [n for n in FIXED_N if n in self.sizes]
        return chosen or self.sizes[-3:]


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


def thread_ticks(ax, values):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Графики ускорения этапа 3.")
    parser.add_argument("-d", "--results-dir", type=Path, default=RESULTS_DIR, help="каталог результатов")
    args = parser.parse_args(argv)
    out = args.results_dir
    m = Measurements(out / "measurements.csv")
    cores = physical_cores(read_platform(out / "platform.txt"))
    parallel = [p for p in m.threads if p > 1]
    colors = plt.cm.tab10.colors  # соседние P должны различаться: P = 8 и P = 12 — самые близкие кривые

    # --- S(N) для каждого P
    for name, func, label in (("speedup_n", m.s_rel, "относительное"), ("speedup_n_abs", m.s_abs, "абсолютное")):
        fig, ax = new_axes(f"Вариант А (директивы OpenMP): {label} ускорение, b = {m.block}", "Ускорение S")
        for p, color in zip(parallel, colors):
            points = [(n, func("block", n, p)) for n in m.sizes if func("block", n, p)]
            beyond = cores is not None and p > cores  # потоков больше, чем ядер: делят ядра (Hyper-Threading)
            ax.plot(*zip(*points), "s--" if beyond else "o-", color=color, linewidth=2.2, markersize=7,
                    label=f"P = {p}" + (" (больше, чем ядер)" if beyond else ""))
        ax.axhline(1.0, color="gray", linewidth=1)
        ax.set_ylim(bottom=0)
        log2_ticks(ax, m.sizes)
        save(fig, ax, out, name)

    for mode in ("block", "phases"):
        header = ["N", "T_посл, с", "T_1, с", *(f"S_отн P={p}" for p in parallel), *(f"S_абс P={p}" for p in parallel)]
        rows = [[n, fmt(m.t_seq(n), 6), fmt(m.t("omp", mode, n, 1), 6),
                 *(fmt(m.s_rel(mode, n, p)) for p in parallel), *(fmt(m.s_abs(mode, n, p)) for p in parallel)]
                for n in m.sizes]
        write_csv(out / f"speedup_{mode}.csv", header, rows)

    # --- S(P) при фиксированных N, идеальное ускорение и закон Амдала
    fixed = m.fixed_sizes()
    fit_threads = [p for p in m.threads if cores is None or p <= cores]
    amdahl_rows = []
    for name, func, label in (("speedup_p", m.s_rel, "относительное"), ("speedup_p_abs", m.s_abs, "абсолютное")):
        fig, ax = new_axes(f"Вариант А: {label} ускорение от числа потоков, b = {m.block}", "Ускорение S",
                           "Число потоков P")
        ax.plot(m.threads, m.threads, ":", color="black", label="идеальное S = P")
        for n, color in zip(fixed, ("#1f77b4", "#ff7f0e", "#2ca02c")):
            s = [func("block", n, p) for p in m.threads]
            ax.plot(m.threads, s, "o-", color=color, linewidth=2.2, markersize=7, label=f"N = {n}")
            if name == "speedup_p":  # закон Амдала — для относительного ускорения
                alpha = amdahl_alpha(fit_threads, [func("block", n, p) for p in fit_threads])
                if alpha is not None and alpha > 0:
                    ax.plot(m.threads, [amdahl_speedup(alpha, p) for p in m.threads], "--", color=color,
                            linewidth=1.2, label=f"закон Амдала, α = {100 * alpha:.1f} %")
        if cores and cores < max(m.threads):
            ax.axvline(cores, color="gray", linestyle="--", linewidth=1.2, label=f"физических ядер: {cores}")
        ax.set_ylim(bottom=0)
        thread_ticks(ax, m.threads)
        save(fig, ax, out, name)

    for mode in ("block", "phases"):
        for n in fixed:
            alpha = amdahl_alpha(fit_threads, [m.s_rel(mode, n, p) for p in fit_threads])
            amdahl_rows.append([mode, n, fmt(alpha, 5), fmt(1 / alpha if alpha and alpha > 0 else None, 1)])
    write_csv(out / "amdahl.csv", ["стратегия", "N", "alpha", "S_max = 1/alpha"], amdahl_rows)

    header = ["P", *(f"{kind} N={n}" for n in fixed for kind in ("S_отн", "S_абс", "E_отн"))]
    rows = [[p, *(v for n in fixed for v in (fmt(m.s_rel("block", n, p)), fmt(m.s_abs("block", n, p)),
                                               fmt(ratio(m.s_rel("block", n, p), p))))] for p in m.threads]
    write_csv(out / "speedup_p.csv", header, rows)

    # --- эффективность
    fig, ax = new_axes(f"Вариант А: эффективность E = S_отн / P, b = {m.block}", "Эффективность E", "Число потоков P")
    model_rows = []
    for n, color in zip(fixed, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        ax.plot(m.threads, [ratio(m.s_rel("block", n, p), p) for p in m.threads], "o-", color=color,
                linewidth=2.2, markersize=7, label=f"N = {n}")
        # модель — только до числа ядер: она считает потоки равноскоростными, а на одном ядре два потока — нет
        ax.plot(fit_threads, [static_schedule_speedup(n, m.block, p, "block") / p for p in fit_threads], "--",
                color=color, linewidth=1.2, label=f"модель распределения блоков, N = {n}")
    for mode in ("block", "phases"):
        for n in fixed:
            for p in fit_threads:
                model = static_schedule_speedup(n, m.block, p, mode)
                model_rows.append([mode, n, p, fmt(model), fmt(m.s_rel(mode, n, p)),
                                   fmt(ratio(m.s_rel(mode, n, p), model))])
    write_csv(out / "model.csv", ["стратегия", "N", "P", "S_модель", "S_отн", "S_отн / S_модель"], model_rows)
    if cores and cores < max(m.threads):
        ax.axvline(cores, color="gray", linestyle="--", linewidth=1.2, label=f"физических ядер: {cores}")
    ax.set_ylim(0, 1.1)
    thread_ticks(ax, m.threads)
    save(fig, ax, out, "efficiency_p")

    # --- стратегии
    p_cmp = max(p for p in m.threads if cores is None or p <= cores)
    fig, ax = new_axes(f"Сравнение стратегий: относительное ускорение при P = {p_cmp}, b = {m.block}", "Ускорение S")
    for mode in ("block", "phases"):
        points = [(n, m.s_rel(mode, n, p_cmp)) for n in m.sizes if m.s_rel(mode, n, p_cmp)]
        ax.plot(*zip(*points), "o-", color=STRATEGY_COLORS[mode], linewidth=2.2, markersize=7,
                label=f"{mode}: {STRATEGY_NAMES[mode]}")
    ax.axhline(1.0, color="gray", linewidth=1)
    ax.set_ylim(bottom=0)
    log2_ticks(ax, m.sizes)
    save(fig, ax, out, "strategies")

    # --- автопараллелизация
    if m.autopar_threads:
        fig, ax = new_axes("Вариант Б: автопараллелизация компилятором (абсолютное ускорение)", "Ускорение S")
        for p, color in zip(m.autopar_threads, ("#2ca02c", "#9467bd", "#8c564b", "#e377c2")):
            points = [(n, m.s_autopar(n, p)) for n in m.sizes if m.s_autopar(n, p)]
            ax.plot(*zip(*points), "o-", color=color, linewidth=2.2, markersize=7, label=f"автопараллелизация, P = {p}")
            omp = [(n, m.s_abs("block", n, p)) for n in m.sizes if m.s_abs("block", n, p)]
            ax.plot(*zip(*omp), ":", color=color, linewidth=1.5, label=f"директивы OpenMP, P = {p}")
        ax.axhline(1.0, color="gray", linewidth=1)
        ax.set_ylim(bottom=0)
        log2_ticks(ax, m.sizes)
        save(fig, ax, out, "autopar")
        write_csv(out / "autopar.csv", ["N", *(f"S_абс P={p}" for p in m.autopar_threads)],
                  [[n, *(fmt(m.s_autopar(n, p)) for p in m.autopar_threads)] for n in m.sizes])

    print(f"Графики и таблицы сохранены в {out}")
    for n in fixed:
        best = max(m.threads, key=lambda p: m.s_rel("block", n, p) or 0)
        print(f"  N = {n}: наибольшее S_отн = {m.s_rel('block', n, best):.2f} при P = {best}, "
              f"S_абс = {m.s_abs('block', n, best):.2f}")


if __name__ == "__main__":
    main()
