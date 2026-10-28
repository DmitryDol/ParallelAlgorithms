"""Поиск собранных программ этапа 2 (общий для сверки, замеров и графиков)."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = Path(os.environ.get("LAB2_BUILD_DIR", HERE / "build"))
RESULTS_DIR = HERE / "results"

# Варианты сборки из CMakeLists.txt: noopt — «без флага», opt — «с флагом».
VARIANTS = ["noopt", "opt", "opt_novec", "opt_fastmath"]


def find_program(variant: str, build_dir: Path = BUILD_DIR):
    """cholesky_<вариант> в каталоге сборки (в т.ч. в Release/ у Visual Studio) или None."""
    if not build_dir.exists():
        return None
    for name in (f"cholesky_{variant}", f"cholesky_{variant}.exe"):
        found = sorted(p for p in build_dir.rglob(name) if p.is_file())
        if found:
            return found[0]
    return None
