"""Поиск собранных программ этапа 3 (общий для сверки, замеров и графиков)."""

import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = Path(os.environ.get("LAB3_BUILD_DIR", HERE / "build"))
RESULTS_DIR = HERE / "results"
# Подробный журнал MSBuild (генератор Visual Studio) в каталоге сборки: в нём сообщения /Qpar-report.
MSBUILD_LOG = "msbuild_detailed.log"


def find_program(variant: str, build_dir: Path = BUILD_DIR):
    """cholesky_<вариант> в каталоге сборки (в т.ч. в Release/ у Visual Studio) или None."""
    if not build_dir.exists():
        return None
    for name in (f"cholesky_{variant}", f"cholesky_{variant}.exe"):
        found = sorted(p for p in build_dir.rglob(name) if p.is_file())
        if found:
            return found[0]
    return None


def autopar_programs(build_dir: Path = BUILD_DIR) -> dict:
    """Сборки с автопараллелизацией: {P: путь} для GCC (autopar_p<P>) или {0: путь} для MSVC (autopar)."""
    found = {}
    if build_dir.exists():
        for path in build_dir.rglob("cholesky_autopar*"):
            m = re.fullmatch(r"cholesky_autopar(?:_p(\d+))?(?:\.exe)?", path.name)
            if m and path.is_file():
                found.setdefault(int(m.group(1) or 0), path)
    return dict(sorted(found.items()))
