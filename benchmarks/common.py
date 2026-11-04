"""Shared helpers for the benchmark scripts: paths, machine description, timing, JSON output."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DATA = Path(os.environ.get("LTR_DATA", ROOT.parent / ".bench-data" / "learning-to-rank"))
MSLR = DATA / "mslr10k"


def machine() -> dict[str, Any]:
    """Hardware and library versions recorded next to every result."""

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "load_average_1_5_15": [round(x, 2) for x in os.getloadavg()],
    }
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        ).stdout
        if brand:
            info["cpu"] = brand
        if mem.strip():
            info["memory_gb"] = round(int(mem) / 2**30, 1)
    except OSError:
        pass
    for mod in ("numpy", "lightgbm", "xgboost", "torch"):
        with contextlib.suppress(ImportError):
            info[mod] = __import__(mod).__version__
    return info


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


def write(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def read(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RESULTS / name).read_text())
    return data
