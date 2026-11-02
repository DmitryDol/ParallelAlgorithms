"""
Изоляция этапов при общем запуске `uv run pytest`.

Каждый этап — копия предыдущего, поэтому вспомогательные модули этапов называются одинаково (programs, bench,
plot_speedup, make_report). Перед импортом каждого тестового модуля каталог его этапа ставится первым в пути
поиска, а одноимённые модули другого этапа убираются из кэша — тогда тест получает модули своего этапа
при любом порядке запуска.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED_NAMES = ("programs", "bench", "plot_speedup", "make_report", "run_all")


def stage_of(path: Path):
    for parent in path.parents:
        if parent.parent == ROOT and parent.name.startswith("lab"):
            return parent
    return None


def pytest_collectstart(collector):
    path = getattr(collector, "path", None)
    stage = stage_of(Path(path)) if path and str(path).endswith(".py") else None
    if stage is None:
        return
    for name in SHARED_NAMES:
        module = sys.modules.get(name)
        if module is not None and stage not in Path(getattr(module, "__file__", "") or ROOT).parents:
            del sys.modules[name]
    if str(stage) in sys.path:
        sys.path.remove(str(stage))
    sys.path.insert(0, str(stage))
