"""
Тесты вспомогательных функций скриптов этапа 2 (bench.py, plot_speedup.py).
    uv run pytest
"""

from bench import choose_block
from plot_speedup import compiler_family, variant_name


def test_compiler_family():
    assert compiler_family("MSVC 195136260") == "msvc"
    assert compiler_family("MSVC 19.51.36260") == "msvc"
    assert compiler_family("GCC 13.3.0") == "gcc"
    assert compiler_family("Clang 18.1.3 (1ubuntu1)") == "gcc"


def test_variant_names_follow_compiler():
    assert variant_name("noopt", "msvc", "/Od") == "без флага"
    assert variant_name("opt_novec", "gcc", "-O3 -march=native -fno-tree-vectorize") == "с флагом, без векторизации"
    assert variant_name("opt_novec", "msvc", "/O2") == "с флагом, без AVX2"
    assert variant_name("opt_fastmath", "gcc", "-O3 -march=native -ffast-math") == "с флагом и -ffast-math"
    assert variant_name("opt_fastmath", "msvc", "/O2 /arch:AVX2 /fp:fast") == "с флагом и /fp:fast"


def test_choose_block_minimizes_mean_relative_time():
    sweep = {1000: {16: 1.10, 24: 1.00, 32: 1.05}, 2000: {16: 9.0, 24: 8.4, 32: 8.0}}
    # относительно лучшего при каждом N: b=16 → (1.10 + 1.125)/2, b=24 → (1.00 + 1.05)/2, b=32 → (1.05 + 1.00)/2
    assert choose_block(sweep) in (24, 32)
    assert choose_block({1000: {8: 2.0, 48: 1.0}}) == 48
    assert choose_block({1000: {8: 1.0, 48: 1.0}}) == 8  # при равенстве — меньший b
