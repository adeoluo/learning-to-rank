from __future__ import annotations

import os
import sys

import pytest

from learning_to_rank.backend import rust_available
from learning_to_rank.data import RankingData, synthetic

BACKENDS = ["python", "rust"] if rust_available() else ["python"]

if os.environ.get("LTR_REQUIRE_RUST") == "1" and not rust_available():
    raise RuntimeError("LTR_REQUIRE_RUST=1 but the Rust extension is not importable")


@pytest.fixture(scope="session")
def small() -> RankingData:
    return synthetic(60, (5, 30), 8, seed=11)


@pytest.fixture(scope="session")
def medium() -> tuple[RankingData, RankingData, RankingData]:
    return (
        synthetic(400, (10, 60), 20, seed=1),
        synthetic(120, (10, 60), 20, seed=2),
        synthetic(200, (10, 60), 20, seed=3),
    )


@pytest.fixture
def openmp_safe() -> None:
    """LightGBM or XGBoost (Homebrew libomp) and PyTorch (bundled libomp) crash when both load in one
    process on macOS; run `make validation` (which never imports torch) there instead."""

    if sys.platform == "darwin" and "torch" in sys.modules:
        pytest.skip(
            "OpenMP libraries after PyTorch in one process crash on macOS; use make validation"
        )
