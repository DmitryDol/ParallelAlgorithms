"""
Тесты генератора черновика отчёта этапа 3: собирается из результатов в results/ и шаблона текста.
    uv run pytest
"""

import html
import re
import zipfile

import pytest

import make_report as report

MSVC_REPORT = """Автопараллелизация: циклы вычислительного ядра (cholesky.cpp), о которых сообщил компилятор

cholesky.cpp:14 (функция dot): C5012: цикл не параллелизован по следующей причине: "1000"
    for (int k = 0; k < len; ++k) s += x[k] * y[k];
cholesky.cpp:40 (функция make_test_matrix): C5011: loop parallelized
    for (int i = 0; i < n; ++i)
cholesky.cpp:46 (функция factor_diagonal_block): C5012: цикл не параллелизован по следующей причине: "506"
    for (int i = 0; i < m; ++i) {
cholesky.cpp:123 (функция cholesky_block): C5012: цикл не параллелизован по следующей причине: "1000"
    for (int ib = jb + 1; ib < p; ++ib) {

Почему остальные циклы не распараллелены (число случаев):
  2 × в теле цикла есть зависимость по данным (код 1000)
  1 × цикл не приводится к канонической форме (код 506)
  (расшифровка кодов — в документации MSVC: «Vectorizer and parallelizer messages»)
"""

GCC_REPORT = """Автопараллелизация: циклы вычислительного ядра (cholesky.cpp), о которых сообщил компилятор

cholesky.cpp:40 (функция make_test_matrix): parallelizing outer loop 1
    for (int i = 0; i < n; ++i)

Почему остальные циклы не распараллелены (число случаев):
  13 × между итерациями есть зависимость по данным (data dependencies exist across iterations)
  8 × компилятор не смог проанализировать зависимости (обращения к одному массиву через указатели) (cannot analyze data dependencies)
"""


def test_number_formatting():
    assert report.dec(7.168) == "7,17"
    assert report.dec_tex(0.89) == "0{,}89"
    assert report.sci(2.275957e-14) == "2,28·10⁻¹⁴"
    assert report.ms(0.0558) == "55,8"
    assert report.ms(3.113) == "3113"
    assert report.percent(0.0448) == "4,5 %"
    assert report.percent(0.00726) == "0,73 %"
    assert report.percent(0.000036) == "0,0036 %"  # b²/N² — не «0,00 %»
    assert report.percent(0.246) == "24,6 %"
    assert report.trend(4.3, 6.9) == "растёт"
    assert report.trend(1.0, 1.05) == "почти не меняется"


def test_words():
    assert report.plural(21, "цикл", "цикла", "циклов") == "21 цикл"
    assert report.plural(3, "цикл", "цикла", "циклов") == "3 цикла"
    assert report.plural(11, "цикл", "цикла", "циклов") == "11 циклов"
    assert report.plural(25, "цикл", "цикла", "циклов") == "25 циклов"
    sizes = [100, 200, 300, 500]
    assert report.n_set([200, 300, 500], sizes) == "при всех $N \\geq 200$"
    assert report.n_set(sizes, sizes) == "при всех $N$"
    assert report.n_set([100, 300], sizes) == "при $N$ = 100 и 300"
    assert report.n_set([], sizes) == "ни при одном $N$"
    assert report.dec_range(0.999, 1.0) == "1,00"  # не «1,00–1,00»
    assert report.dec_range(0.69, 0.87) == "0,69–0,87"
    assert report.points([(100, 24)]) == "при $N = 100$, $P = 24$"
    assert report.points([(100, 16), (100, 24)]) == "при $N = 100$ и $P$ = 16 и 24"
    assert report.points([(100, 24), (200, 24)]) == "при $P = 24$ и $N$ = 100 и 200"


def test_hyperthreading_dip():
    # при 16 потоках на 12 ядрах ускорение ниже, чем и при 12, и при 24: часть ядер занята двумя потоками
    assert report.hyperthreading_dip({12: 10.5, 16: 6.7, 24: 8.4}, cores=12) == 16
    assert report.hyperthreading_dip({12: 6.2, 16: 7.2, 24: 5.1}, cores=12) is None
    assert report.hyperthreading_dip({12: 10.5, 24: 8.4}, cores=12) is None
    assert report.hyperthreading_dip({4: 3.5, 6: 2.0, 8: 3.0}, cores=4) == 6
    assert report.hyperthreading_dip({4: 3.5, 6: 2.0, 7: 3.0}, cores=4) is None  # 7 ≠ 2 × ядер


def test_openmp_version_text():
    assert report.openmp_text("200203") == "OpenMP 2.0"
    assert report.openmp_text("201511") == "OpenMP 4.5"
    assert report.openmp_text("209999") == "OpenMP (_OPENMP = 209999)"
    assert report.openmp_text("") == "—"


def test_table_threads():
    assert report.table_threads([1, 2, 3, 4, 6, 8, 12, 16, 24], 12) == [2, 4, 8, 12, 24]
    assert report.table_threads([1, 2, 3, 4], 4) == [2, 4]
    assert report.table_threads([1, 2, 3, 4, 6, 8], None) == [2, 4, 8]


def test_agreement_limit():
    # наибольшее P, до которого (включительно) все замеры не ниже 90 % от модели
    ratios = {2: [1.0, 0.99], 4: [0.98], 8: [0.93, 0.95], 12: [0.67, 0.9]}
    assert report.agreement_limit(ratios) == 8
    assert report.agreement_limit({2: [0.85], 4: [0.99]}) is None
    assert report.agreement_limit({2: [1.02], 4: [1.05]}) == 4


def test_parse_autopar_msvc():
    r = report.parse_autopar(MSVC_REPORT)
    assert r["found"] and r["compiler_lines"]
    assert r["loops"] == [(14, "dot", "1000"), (40, "make_test_matrix", None), (46, "factor_diagonal_block", "506"),
                          (123, "cholesky_block", "1000")]
    assert r["parallelized"] == [(40, "make_test_matrix")]
    assert r["reasons"] == [(2, "в теле цикла есть зависимость по данным (код 1000)"),
                            (1, "цикл не приводится к канонической форме (код 506)")]


def test_parse_autopar_gcc():
    r = report.parse_autopar(GCC_REPORT)
    assert r["found"] and not r["compiler_lines"]  # GCC сообщает только распараллеленные циклы
    assert r["parallelized"] == [(40, "make_test_matrix")]
    assert r["reasons"][0][0] == 13 and r["reasons"][1][0] == 8


def test_parse_autopar_missing():
    r = report.parse_autopar("Отчёт компилятора об автопараллелизации не найден (нужна сборка через run_all.py).\n")
    assert not r["found"] and r["parallelized"] == [] and r["reasons"] == []


def test_decimal_comma_in_formulas_is_one_run():
    # «1{,}5» pandoc разбил бы на «1», «,», «5», и Word мог бы поставить пробел после запятой
    assert report.decimal_commas_as_text(r"\frac{1{,}5\, b}{N} = 0{,}89") == r"\frac{\text{1,5}\, b}{N} = \text{0,89}"
    assert report.decimal_commas_as_text(r"i, j = 0, \ldots") == r"i, j = 0, \ldots"


def test_unknown_placeholder_is_reported():
    with pytest.raises(SystemExit):
        report.substitute("текст <<нет_такого>>", {})


def test_report_is_built_from_results(tmp_path):
    if not (report.RESULTS_DIR / "measurements.csv").exists():
        pytest.skip("нет results/measurements.csv — сначала bench.py")
    output = tmp_path / "report.docx"
    report.build(output=output)

    with zipfile.ZipFile(output) as docx:
        document = docx.read("word/document.xml").decode("utf-8")
        media = [name for name in docx.namelist() if name.startswith("word/media/")]
    text = html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", document)))  # «&lt;&lt;» → «<<»
    template = report.TEMPLATE_TEXT.read_text(encoding="utf-8")
    figures = len(re.findall(r"^!figure ", template, re.M))
    tables = len(re.findall(r"^!table ", template, re.M))

    assert "ЛАБОРАТОРНАЯ РАБОТА № 3" in text and "Лабораторная работа № 3" in text
    assert "РАБОТА № 1" not in text and "Лабораторная работа № 1" not in text  # заголовки от отчёта 1 заменены
    assert "<<" not in text and ">>" not in text
    assert "None" not in text and "nan" not in text  # нет пропущенных чисел
    assert document.count("<w:drawing>") == figures == len(media)
    assert all(f"Таблица {k}." in text for k in range(1, tables + 1))
    assert all(f"Рисунок {k} –" in text for k in range(1, figures + 1))
    assert document.count("<m:oMath>") > 50          # формулы — объекты Word, а не текст
    assert "$" not in text                           # LaTeX не остался в тексте
    assert "#pragma omp parallel for schedule(static)" in text  # листинг директив


def write_gcc_results(path, sizes=(100, 200, 300, 500), threads=(1, 2, 4), b=16):
    """Синтетические результаты GCC (4 ядра, без больших N): время — по модели распределения блоков."""
    import csv

    from plot_speedup import static_schedule_speedup

    flags = {"seq": "-O3 -march=native", "omp": "-O3 -march=native -fopenmp",
             "autopar": "-O3 -march=native -ftree-parallelize-loops=4"}
    rows = []

    def add(experiment, mode, n, p, t):
        rows.append({"experiment": experiment, "variant": experiment, "flags": flags[experiment],
                     "compiler": "GCC 13.3.0", "openmp": "201511" if experiment == "omp" else "", "mode": mode,
                     "N": n, "b": b, "threads": p, "repeats": 3, "batch": 1, "t_min": t, "t_median": t * 1.01,
                     "gflops": n**3 / 3 / t / 1e9, "err_L": 1e-17 * n})

    for n in sizes:
        t1 = n**3 / 3 / 3e9
        add("seq", "block", n, 1, t1)
        for p in threads:
            for mode in ("block", "phases"):
                slower = {1: 1.0, 2: 1.05}.get(p, 1.3)  # при P = 4 — ниже модели, как в контейнере
                add("omp", mode, n, p, t1 / static_schedule_speedup(n, b, p, mode) * slower)
        for p in (2, 4):
            add("autopar", "block", n, p, t1)
    with open(path / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "platform.txt").write_text(
        "Процессор: Intel(R) Xeon(R) Processor\nФизических ядер: 4\nЛогических процессоров: 4\nКэш L3: 105 МБ\n"
        f"ОС: Linux\nРазмер блока: {b} (задан)\n\nСборки:\n", encoding="utf-8")
    (path / "autopar_report.txt").write_text(GCC_REPORT, encoding="utf-8")


def test_report_from_gcc_results_without_large_n(tmp_path):
    import plot_speedup

    write_gcc_results(tmp_path)
    plot_speedup.main(["-d", str(tmp_path)])
    output = tmp_path / "report.docx"
    report.build(results_dir=tmp_path, output=output)
    with zipfile.ZipFile(output) as docx:
        document = docx.read("word/document.xml").decode("utf-8")
    text = html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", document)))
    assert "-ftree-parallelize-loops=P" in text and "cholesky_autopar_p<P>" in text
    assert "/Qpar" not in text
    assert "распараллелил цикл в функции make_test_matrix (строка 40)" in text
    assert "в самом разложении не распараллелен ни один цикл" in text
    assert "<<" not in text and "None" not in text
