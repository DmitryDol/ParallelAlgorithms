"""Поиск собранных программ этапа 4 и запуск MPI-программы (общий для сверки, замеров и графиков)."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = Path(os.environ.get("LAB4_BUILD_DIR", HERE / "build"))
RESULTS_DIR = HERE / "results"


def find_program(variant: str, build_dir: Path = BUILD_DIR):
    """cholesky_<вариант> (seq или mpi) в каталоге сборки (в т.ч. в Release/ у Visual Studio) или None."""
    if not build_dir.exists():
        return None
    for name in (f"cholesky_{variant}", f"cholesky_{variant}.exe"):
        found = sorted(p for p in build_dir.rglob(name) if p.is_file())
        if found:
            return found[0]
    return None


def find_mpiexec(build_dir: Path = BUILD_DIR):
    """mpiexec, найденный CMake при настройке (MPIEXEC_EXECUTABLE в CMakeCache.txt), иначе из PATH; или None."""
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        m = re.search(r"^MPIEXEC_EXECUTABLE:FILEPATH=(.+)$", cache.read_text(encoding="utf-8", errors="replace"),
                      re.MULTILINE)
        if m and Path(m.group(1).strip()).exists():
            return m.group(1).strip()
    return shutil.which("mpiexec") or shutil.which("mpirun")


def mpi_family(version: str) -> str:
    """Реализация MPI по строке mpi= из `cholesky_mpi --info`."""
    if version.startswith("Open MPI"):
        return "openmpi"
    if version.startswith("MPICH"):
        return "mpich"
    if version.startswith("Microsoft MPI"):
        return "msmpi"
    return "other"


def launch_command(mpiexec: str, procs: int, program: Path, args: list, family: str, cores: int) -> list:
    """mpiexec -n P программа аргументы. Open MPI по умолчанию не запускает процессов больше, чем ядер, а с
    --oversubscribe перестаёт привязывать процессы к ядрам — поэтому ключ только тогда, когда процессов больше."""
    extra = ["--oversubscribe"] if family == "openmpi" and procs > cores else []
    return [mpiexec, "-n", str(procs), *extra, str(program), *map(str, args)]


def launch_env(family: str, is_root=None) -> dict:
    """Open MPI не запускает процессы от root без явного разрешения (так бывает только в облачном контейнере)."""
    if is_root is None:
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if family == "openmpi" and is_root:
        return {"OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"}
    return {}


def program_info(program: Path, mpiexec=None) -> dict:
    """Строки key=value из `программа --info` (MPI-программа — под mpiexec -n 1)."""
    command = [str(program), "--info"]
    env = {**os.environ}
    if mpiexec:
        command = [mpiexec, "-n", "1", *command]
        env.update(launch_env("openmpi"))  # реализация ещё неизвестна; другим MPI эти переменные не мешают
    out = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if out.returncode != 0:
        sys.exit(f"Ошибка: {' '.join(command)} завершилась с кодом {out.returncode}\n{out.stderr}")
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
