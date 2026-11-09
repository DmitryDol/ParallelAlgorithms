"""
Тесты генератора черновика отчёта этапа 4: собирается из результатов в results/ и шаблона текста.
    uv run pytest
"""

import csv
import html
import re
import zipfile

import pytest

import make_report as report


def document_text(path):
    with zipfile.ZipFile(path) as docx:
        document = docx.read("word/document.xml").decode("utf-8")
        media = [name for name in docx.namelist() if name.startswith("word/media/")]
    return document, html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", document))), media


def test_number_formatting():
    assert report.dec(7.168) == "7,17"
    assert report.sci(2.275957e-14) == "2,28·10⁻¹⁴"
    assert report.ms(0.0558) == "55,8"
    assert report.ms(3.113) == "3113"
    assert report.percent(0.0448) == "4,5 %"
    assert report.percent(0.000036) == "0,0036 %"
    assert report.mb(36_000_000) == "34 МБ"
    assert report.mb(450_000) == "0,43 МБ"


def test_words():
    assert report.plural(21, "раз", "раза", "раз") == "21 раз"
    assert report.plural(3, "процесс", "процесса", "процессов") == "3 процесса"
    sizes = [100, 200, 300, 500]
    assert report.n_set([200, 300, 500], sizes) == "при всех $N \\geq 200$"
    assert report.n_set([100, 300], sizes) == "при $N$ = 100 и 300"
    assert report.n_set([500], sizes) == "при $N$ = 500"  # одно значение — не «при всех N ≥ 500»
    assert report.times_slower(32.79) == "в 33 раза"
    assert report.times_slower(1.94) == "в 1,9 раза"
    assert report.times_slower(25.6) == "в 26 раз"
    assert report.times_slower(21.2) == "в 21 раз"
    assert report.times_slower(12.4) == "в 12 раз"
    # диапазон — только для подряд идущих размерностей
    assert report.n_range([100], sizes) == "$N = 100$"
    assert report.n_range([100, 200, 300], sizes) == "$N = 100 \\ldots 300$"
    assert report.n_range([100, 500], sizes) == "$N$ = 100 и 500"


def test_table_procs():
    assert report.table_procs([1, 2, 3, 4, 6, 8, 12, 16, 24], 12) == [2, 4, 8, 12, 16, 24]
    assert report.table_procs([1, 2, 3, 4], 4) == [2, 4]


def test_agreement_limit():
    assert report.agreement_limit({2: [1.0], 4: [0.93], 6: [0.87, 0.96], 8: [0.83]}) == 4
    assert report.agreement_limit({2: [0.85], 4: [0.99]}) is None


def test_broadcast_volume():
    # на шаге j рассылается строка j: mj × (j0 + mj) чисел — в обеих стратегиях одно и то же число чисел
    assert report.broadcast_numbers(24, 12) == 12 * 12 + 12 * 24
    assert report.broadcast_numbers(30, 12) == 12 * 12 + 12 * 24 + 6 * 30  # последний блок неполный
    assert report.broadcast_numbers(3000, 12) == pytest.approx(3000**2 / 2, rel=0.01)


def test_conditional_blocks():
    text = "a\n!if omp\nb\n!endif\nc\n!if other\nd\n!endif\n"
    assert report.apply_conditions(text, {"omp": True, "other": False}) == "a\nb\nc\n"
    with pytest.raises(SystemExit):
        report.apply_conditions("!if unknown\nx\n!endif\n", {})
    with pytest.raises(SystemExit):
        report.apply_conditions("!if omp\nx\n", {"omp": True})  # нет !endif


def test_decimal_comma_in_formulas_is_one_run():
    assert report.decimal_commas_as_text(r"\frac{1{,}5\, b}{N}") == r"\frac{\text{1,5}\, b}{N}"


def test_unknown_placeholder_is_reported():
    with pytest.raises(SystemExit):
        report.substitute("текст <<нет_такого>>", {})


def test_report_is_built_from_results(tmp_path):
    if not (report.RESULTS_DIR / "measurements.csv").exists():
        pytest.skip("нет results/measurements.csv — сначала bench.py")
    output = tmp_path / "report.docx"
    report.build(output=output)
    document, text, media = document_text(output)
    template = report.TEMPLATE_TEXT.read_text(encoding="utf-8")
    tables = len(re.findall(r"^!table ", template, re.M))

    assert "ЛАБОРАТОРНАЯ РАБОТА № 4" in text and "Лабораторная работа № 4" in text
    assert "РАБОТА № 1" not in text and "Лабораторная работа № 1" not in text
    assert "<<" not in text and ">>" not in text and "!if" not in text
    assert "None" not in text and "nan" not in text
    figures = document.count("<w:drawing>")
    assert figures == len(media) >= 6
    assert all(f"Рисунок {k} –" in text for k in range(1, figures + 1))
    assert sum(f"Таблица {k}." in text for k in range(1, tables + 1)) >= tables - 1  # одна таблица — по условию
    assert document.count("<m:oMath>") > 50
    assert "$" not in text
    assert "MPI_Bcast" in text


def write_openmpi_results(path, sizes=(100, 200, 300, 500), procs=(1, 2, 4), b=16):
    """Синтетические результаты Open MPI + GCC (4 ядра, без больших N, без этапа 3): время — по модели."""
    from plot_speedup import cyclic_model_speedup

    rows = []

    def add(experiment, mode, n, p, t):
        rows.append({"experiment": experiment, "variant": experiment, "flags": "-O3 -march=native",
                     "compiler": "GCC 13.3.0", "mpi": "Open MPI v4.1.6 (MPI 3.1)" if experiment == "mpi" else "",
                     "mode": mode, "N": n, "b": b, "procs": p, "repeats": 3, "batch": 1, "t_min": t,
                     "t_median": t * 1.01, "gflops": n**3 / 3 / t / 1e9, "err_L": 1e-17 * n})

    for n in sizes:
        t1 = n**3 / 3 / 3e9
        add("seq", "block", n, 1, t1)
        for p in procs:
            for mode in ("block", "phases"):
                slower = {1: 1.0, 2: 1.05}.get(p, 1.3)  # при P = 4 — ниже модели
                add("mpi", mode, n, p, t1 / cyclic_model_speedup(n, b, p, mode) * slower)
    with open(path / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "platform.txt").write_text(
        "Процессор: Intel(R) Xeon(R) Processor\nФизических ядер: 4\nЛогических процессоров: 4\nКэш L3: 105 МБ\n"
        f"ОС: Linux\nРеализация MPI: Open MPI v4.1.6 (MPI 3.1)\nРазмер блока: {b} (задан)\n\nСборки:\n",
        encoding="utf-8")


def test_report_from_openmpi_results_without_openmp_stage(tmp_path):
    import plot_speedup

    write_openmpi_results(tmp_path)
    plot_speedup.main(["-d", str(tmp_path), "--openmp-dir", str(tmp_path / "нет")])
    assert not (tmp_path / "mpi_vs_omp.png").exists()
    output = tmp_path / "report.docx"
    report.build(results_dir=tmp_path, output=output)
    _, text, media = document_text(output)
    assert "Open MPI v4.1.6" in text and "-O3 -march=native" in text
    assert "MS-MPI" not in text
    assert "<<" not in text and "None" not in text and "!if" not in text
    assert len(media) == 6  # без рисунка сравнения с OpenMP
    assert "Сравнение с OpenMP не выполнялось" in text


USER_PROCS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
USER_SIZES = (100, 200, 300, 500, 1000, 2000, 3000)


def write_user_like_results(path, omp_reaches_model, b=12):
    """Синтетика «как у пользователя» (12 ядер / 24 логических): MPI отстаёт от модели тем сильнее, чем больше P
    и меньше N (обмены), замеры до 12 процессов устойчивы; при P = 24 — обвал отдельных точек (block, N = 100 и 500;
    phases, N = 200), при остальных малых N его нет. OpenMP (этап 3) — на том же процессоре: либо достигает модели
    до 12 потоков, либо при 12 потоках упирается в 65 % от неё."""
    from plot_speedup import cyclic_model_speedup

    mpi_dir, omp_dir = path / "mpi", path / "omp"
    mpi_dir.mkdir()
    omp_dir.mkdir()
    platform = "Процессор: Intel(R) Xeon(R) CPU E5-2678 v3\nФизических ядер: 12\nЛогических процессоров: 24\n"
    rows, omp_rows = [], []
    collapse = {("block", 100): 0.04, ("block", 500): 0.75, ("phases", 200): 0.13}

    def add(experiment, mode, n, p, t, spread):
        rows.append({"experiment": experiment, "variant": experiment, "flags": "/O2", "compiler": "MSVC 19.51",
                     "mpi": "MS-MPI 10.1 (MPI 2.0)" if experiment == "mpi" else "", "mode": mode, "N": n, "b": b,
                     "procs": p, "repeats": 5, "batch": 1, "t_min": t, "t_median": t * spread,
                     "gflops": n**3 / 3 / t / 1e9, "err_L": 1e-17 * n})

    for n in USER_SIZES:
        t1 = n**3 / 3 / 3e9
        add("seq", "block", n, 1, t1, 1.0)
        for mode in ("block", "phases"):
            cost = {"block": 0.01, "phases": 0.018}[mode]  # обмены: доля растёт с P и убывает с N
            s12 = cyclic_model_speedup(n, b, 12, mode) / (1 + cost * 11 * 1000 / n)
            for p in USER_PROCS:
                if p <= 12:
                    s, spread = cyclic_model_speedup(n, b, p, mode) / (1 + cost * (p - 1) * 1000 / n), 1.01
                else:
                    s, spread = s12 * (0.75 if p == 16 else 0.92), 1.06
                if p == 24 and (mode, n) in collapse:
                    s, spread = collapse[mode, n], 1.3
                add("mpi", mode, n, p, t1 / s, spread)
                omp_s = cyclic_model_speedup(n, b, min(p, 12), mode) * (1.0 if p <= 12 else 0.7)
                if not omp_reaches_model and p >= 8:
                    omp_s *= 0.65
                omp_rows.append({"experiment": "omp", "mode": mode, "N": n, "threads": p, "b": b,
                                 "t_min": t1 / omp_s, "err_L": 1e-17 * n})
    for directory, data in ((mpi_dir, rows), (omp_dir, omp_rows)):
        with open(directory / "measurements.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        (directory / "platform.txt").write_text(platform, encoding="utf-8")
    (mpi_dir / "mpi_vs_omp.png").write_bytes(b"")  # рисунок сравнения есть — сравнение войдёт в отчёт
    return mpi_dir, omp_dir


def user_like_analysis(tmp_path, omp_reaches_model):
    from plot_speedup import Measurements

    mpi_dir, omp_dir = write_user_like_results(tmp_path, omp_reaches_model)
    m = Measurements(mpi_dir / "measurements.csv")
    values, _, flags = report.compute(m, report.read_platform(mpi_dir / "platform.txt"), mpi_dir, omp_dir)
    assert flags["omp"]
    return values


def test_gap_is_attributed_to_exchanges_when_openmp_reaches_model(tmp_path):
    # OpenMP на том же компьютере достигает модели до 12 потоков — отставание MPI нельзя списывать на компьютер
    v = user_like_analysis(tmp_path, omp_reaches_model=True)
    many, omp, conclusions = v["analysis_many"], v["analysis_omp"], v["analysis_conclusions"]
    assert "цена обменов" in many
    assert "упирается" not in many and "не только обмены" not in many and "частоты" not in many
    assert "упираются" not in omp and "в пределах разброса" not in omp
    assert "ограничивает уже компьютер" not in conclusions and "цена обменов" in conclusions
    # замеры при P = 12 устойчивы — стратегии сравниваются на всех ядрах, без оговорки о разбросе
    assert "При $P = 12$ стратегия" in v["analysis_strategies"] and "неустойчив" not in v["analysis_strategies"]
    # при 16 процессах на 12 ядрах — провал (часть ядер с двумя процессами), при 24 — снова равномерно
    assert "провала при $P = 16$" in many
    # обвал при P = 24 — в отдельных точках, а при N = 200 у block его нет
    assert "эпизодическ" in many
    assert "100 \\ldots 500" not in v["analysis_n"] and "$N$ = 100 и 500" in v["analysis_n"]


def test_gap_is_attributed_to_computer_when_openmp_falls_short_too(tmp_path):
    v = user_like_analysis(tmp_path, omp_reaches_model=False)
    assert "упирается в то же ограничение" in v["analysis_many"]
    assert "цена обменов" not in v["analysis_many"]


def test_scheme_is_drawn(tmp_path):
    import plot_speedup

    plot_speedup.draw_scheme(tmp_path)
    assert (tmp_path / "scheme.png").stat().st_size > 10_000


def test_scheme_legend_does_not_cover_blocks():
    # у пользователя matplotlib рисует шрифтом DejaVu Sans — он шире, и легенда наезжала на последний блок
    import matplotlib
    from matplotlib.patches import Rectangle

    import plot_speedup

    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax, legend = plot_speedup.scheme_figure()
        fig.canvas.draw()
        box = legend.get_window_extent()
        blocks = [p.get_window_extent() for p in ax.patches if isinstance(p, Rectangle)]
        assert blocks and all(not box.overlaps(b) for b in blocks)

