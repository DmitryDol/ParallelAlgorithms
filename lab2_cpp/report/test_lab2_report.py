"""
Тесты генератора черновика отчёта этапа 2: собирается из результатов в results/ и шаблона текста.
    uv run pytest
"""

import re
import zipfile

import pytest

import make_report as report


def test_number_formatting():
    assert report.dec(7.168) == "7,17"
    assert report.dec_tex(0.89) == "0{,}89"
    assert report.sci(2.275957e-14) == "2,28·10⁻¹⁴"
    assert report.sci(9.999e-16) == "1,00·10⁻¹⁵"
    assert report.ms(0.0558) == "55,8"
    assert report.ms(3.113) == "3113"
    assert report.ms(1.27e-5) == "0,0127"
    assert report.trend(4.3, 6.9) == "растёт"
    assert report.trend(7.1, 3.7) == "уменьшается"
    assert report.trend(1.0, 1.05) == "почти не меняется"
    assert report.per_core_kb("8 МБ (4 × 2 МБ)") == 2048
    assert report.per_core_kb("192 КБ (4 × 48 КБ)") == 48
    assert report.per_core_kb("480 МБ") == 480 * 1024
    assert report.per_core_kb(None) is None


def test_compiler_text():
    assert report.compiler_text("MSVC 195136260") == "MSVC 19.51.36260"
    assert report.compiler_text("MSVC 19.51.36260") == "MSVC 19.51.36260"
    assert report.compiler_text("GCC 13.3.0") == "GCC 13.3.0"


def test_times_phrase():
    assert report.times_phrase(1.7) == "быстрее в 1,70 раза"
    assert report.times_phrase(0.95) == "медленнее в 1,05 раза"


def test_n_equals():
    assert report.n_equals([2048]) == "$N = 2048$"
    assert report.n_equals([1024, 2048]) == "$N = 1024$ и $N = 2048$"
    assert report.n_equals([256, 1024, 2048]) == "$N = 256$, $N = 1024$ и $N = 2048$"


def test_trend_text():
    assert report.trend_text(4.33, 64, 7.08, 1536) == "растёт с 4,33 при $N = 64$ до 7,08 при $N = 1536$"
    assert report.trend_text(4.26, 64, 3.89, 1536) == "почти не меняется: 4,26 при $N = 64$ и 3,89 при $N = 1536$"


def test_power_of_two_dips():
    sizes = [512, 768, 1024, 1536, 2048]
    assert report.power_of_two_dips(sizes, [2.3, 2.3, 2.05, 2.2, 1.6]) == [1024, 2048]
    assert report.power_of_two_dips(sizes, [6.3, 6.5, 6.5, 6.6, 6.4]) == []
    assert report.power_of_two_dips([64, 96, 128], [1.0, 1.0, 0.5]) == [128]


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
    text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", document))
    template = report.TEMPLATE_TEXT.read_text(encoding="utf-8")
    figures = len(re.findall(r"^!figure ", template, re.M))
    tables = len(re.findall(r"^!table ", template, re.M))

    assert "ЛАБОРАТОРНАЯ РАБОТА № 2" in text and "Лабораторная работа № 2" in text
    assert "РАБОТА № 1" not in text and "Лабораторная работа № 1" not in text  # заголовки от отчёта 1 заменены
    assert "<<" not in text and ">>" not in text
    assert document.count("<w:drawing>") == figures == len(media)
    assert all(f"Таблица {k}." in text for k in range(1, tables + 1))
    assert all(f"Рисунок {k} –" in text for k in range(1, figures + 1))
    assert document.count("<m:oMath>") > 50          # формулы — объекты Word, а не текст
    assert "$" not in text                           # LaTeX не остался в тексте
