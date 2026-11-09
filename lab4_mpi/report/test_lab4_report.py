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

