"""
Этап 2: таблицы и графики ускорения по results/measurements.csv (его пишет bench.py).

Запуск из корня репозитория:
    uv run python lab2_cpp/plot_speedup.py

Ускорение S = T_базовый / T_улучшенный считается по минимальному времени из повторов.
Результаты (в results/):
  speedup_flags        — вариант А: без флага → с флагом (блочный и точечный алгоритмы);
  speedup_block        — вариант Б: точечный → блочный (для каждой сборки);
  speedup_vectorization — вклад векторизации (GCC: без неё → с ней; MSVC: без AVX2 → с AVX2) и ключа,
                          разрешающего переставлять операции (-ffast-math или /fp:fast);
  time_vs_n, gflops_vs_n, time_vs_block — время, производительность и подбор размера блока.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from programs import RESULTS_DIR

VARIANTS_ORDER = ["noopt", "opt", "opt_novec", "opt_fastmath"]
MODE_NAMES = {"block": "блочный", "point": "точечный"}
COLORS = {"noopt": "#d62728", "opt": "#1f77b4", "opt_novec": "#2ca02c", "opt_fastmath": "#9467bd"}


def compiler_family(compiler: str) -> str:
    """«msvc» для Visual Studio, «gcc» для GCC и Clang (у них одинаковые ключи)."""
    return "msvc" if compiler.startswith("MSVC") else "gcc"


def variant_name(variant: str, family: str, flags: str) -> str:
    """Название сборки для подписей: одно место для графиков и отчёта."""
    if variant == "noopt":
        return "без флага"
    if variant == "opt":
        return "с флагом"
    if variant == "opt_novec":
        return "с флагом, без AVX2" if family == "msvc" else "с флагом, без векторизации"
    return f"с флагом и {flags.split()[-1]}"  # opt_fastmath: -ffast-math или /fp:fast


class Measurements:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.time = {}  # (эксперимент, сборка, режим, N, b) → t_min
        self.error = {}  # (эксперимент, сборка, режим, N, b) → max |L − L_точн|
        self.flags, self.compiler = {}, {}
        for r in rows:
            b = int(r["b"]) if r["b"] else None
            key = (r["experiment"], r["variant"], r["mode"], int(r["N"]), b)
            self.time[key] = float(r["t_min"])
            self.error[key] = float(r["err_L"])
            self.flags[r["variant"]] = r["flags"]
            self.compiler[r["variant"]] = r["compiler"]
        self.repeats = min(int(r["repeats"]) for r in rows)  # наименьшее число замеров на точку
        self.sizes = sorted({key[3] for key in self.time if key[0] == "sizes"})
        self.block = next(key[4] for key in self.time if key[0] == "sizes" and key[2] == "block")
        self.variants = [v for v in VARIANTS_ORDER if v in self.flags]
        self.family = compiler_family(self.compiler.get("opt", ""))

    def name(self, variant):
        return variant_name(variant, self.family, self.flags[variant])

    def key(self, variant, mode, n, experiment="sizes", b=None):
        if mode == "block" and b is None:
            b = self.block
        return experiment, variant, mode, n, b if mode == "block" else None

    def t(self, variant, mode, n, experiment="sizes", b=None):
        return self.time.get(self.key(variant, mode, n, experiment, b))

    def err(self, variant, mode, n, experiment="sizes", b=None):
        return self.error.get(self.key(variant, mode, n, experiment, b))

    def label(self, variant):
        return f"{self.name(variant)} ({self.flags[variant]})"


def ratio(numerator, denominator):
    return numerator / denominator if numerator and denominator else None


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


def save(fig, ax, out_dir, name, legend=True):
    if legend:
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def speedup_plot(m, out_dir, name, title, curves):
    """curves: [(подпись, цвет, стиль, функция N → ускорение)] — график и таблица CSV."""
    fig, ax = new_axes(title, "Ускорение S")
    table = {n: {} for n in m.sizes}
    for label, color, style, func in curves:
        points = [(n, func(n)) for n in m.sizes]
        points = [(n, s) for n, s in points if s is not None]
        if not points:
            continue
        ax.plot(*zip(*points), style, color=color, linewidth=2.5, markersize=8, label=label)
        for n, s in points:
            table[n][label] = s
    ax.axhline(1.0, color="gray", linewidth=1)
    ax.set_ylim(bottom=0)
    log2_ticks(ax, m.sizes)
    save(fig, ax, out_dir, name)

    labels = [c[0] for c in curves if any(c[0] in row for row in table.values())]
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", *labels])
        for n in m.sizes:
            writer.writerow([n, *(f"{table[n][l]:.3f}" if l in table[n] else "" for l in labels)])
    print(f"\n{title}")
    for n in m.sizes:
        print(f"  N = {n:5d}: " + "; ".join(f"{l}: {table[n][l]:.2f}" for l in labels if l in table[n]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Графики ускорения этапа 2.")
    parser.add_argument("-d", "--results-dir", type=Path, default=RESULTS_DIR, help="каталог результатов")
    args = parser.parse_args(argv)
    out = args.results_dir
    m = Measurements(out / "measurements.csv")

    speedup_plot(m, out, "speedup_flags",
                 "Вариант А: ускорение от флага компиляции",
                 [(f"{MODE_NAMES[mode]} алгоритм: {m.flags['noopt']} → {m.flags['opt']}", COLORS["opt"],
                   "o-" if mode == "block" else "s--",
                   lambda n, mode=mode: ratio(m.t("noopt", mode, n), m.t("opt", mode, n)))
                  for mode in ("block", "point")])

    speedup_plot(m, out, "speedup_block",
                 f"Вариант Б: ускорение блочного алгоритма (b = {m.block}) относительно точечного",
                 [(m.label(v), COLORS[v], "o-" if v in ("noopt", "opt") else "o--",
                   lambda n, v=v: ratio(m.t(v, "point", n), m.t(v, "block", n)))
                  for v in m.variants])

    vector_curves = []
    for mode in ("block", "point"):
        style = "o-" if mode == "block" else "s--"
        if "opt_novec" in m.variants:
            what = "без AVX2 → с AVX2" if m.family == "msvc" else "без векторизации → с ней"
            vector_curves.append((f"{MODE_NAMES[mode]}: {what}", COLORS["opt_novec"], style,
                                  lambda n, mode=mode: ratio(m.t("opt_novec", mode, n), m.t("opt", mode, n))))
        if "opt_fastmath" in m.variants:
            vector_curves.append((f"{MODE_NAMES[mode]}: с флагом → + {m.flags['opt_fastmath'].split()[-1]}",
                                  COLORS["opt_fastmath"], style,
                                  lambda n, mode=mode: ratio(m.t("opt", mode, n), m.t("opt_fastmath", mode, n))))
    if vector_curves:
        speedup_plot(m, out, "speedup_vectorization", "Вклад векторизации", vector_curves)

    for name, ylabel, value in (
        ("time_vs_n", "Время разложения, с", lambda t, n: t),
        ("gflops_vs_n", "Производительность, GFLOP/s", lambda t, n: n**3 / 3 / t / 1e9),
    ):
        fig, ax = new_axes("Время разложения" if name == "time_vs_n" else "Производительность (N³/3 операций)",
                           ylabel)
        fig.set_size_inches(10, 8.5)
        for v in m.variants:
            for mode in ("block", "point"):
                points = [(n, value(m.t(v, mode, n), n)) for n in m.sizes if m.t(v, mode, n)]
                ax.plot(*zip(*points), "o-" if mode == "block" else "s--", color=COLORS[v], linewidth=2,
                        markersize=6, label=f"{MODE_NAMES[mode]}, {m.name(v)}")
        if name == "time_vs_n":
            ax.set_yscale("log")
            n0, t0 = m.sizes[-1], m.t("opt", "block", m.sizes[-1])
            ax.plot(m.sizes, [t0 * (n / n0) ** 3 for n in m.sizes], ":", color="black", label="∝ N³")
        else:
            ax.set_ylim(bottom=0)
        log2_ticks(ax, m.sizes)
        ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))  # под графиком
        save(fig, ax, out, name, legend=False)

    sweep = defaultdict(dict)
    for (experiment, variant, _, n, b), t in m.time.items():
        if experiment == "block_sweep":
            sweep[n][b] = t
    if sweep:
        fig, ax = new_axes("Подбор размера блока (сборка с флагом)", "Время разложения, с", "Размер блока b")
        for n in sorted(sweep):
            blocks = sorted(sweep[n])
            best = min(blocks, key=lambda b: sweep[n][b])
            ax.plot(blocks, [sweep[n][b] for b in blocks], "o-", linewidth=2.5, markersize=8,
                    label=f"N = {n} (минимум при b = {best})")
            print(f"\nПодбор блока, N = {n}: лучший b = {best}, t = {sweep[n][best]:.4f} с")
        ax.axvline(m.block, color="gray", linestyle="--", linewidth=1.5, label=f"используемый b = {m.block}")
        log2_ticks(ax, sorted({b for n in sweep for b in sweep[n]}))
        ax.set_ylim(bottom=0)
        save(fig, ax, out, "time_vs_block")

    print(f"\nГрафики и таблицы сохранены в {out}")


if __name__ == "__main__":
    main()
