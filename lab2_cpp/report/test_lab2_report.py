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
