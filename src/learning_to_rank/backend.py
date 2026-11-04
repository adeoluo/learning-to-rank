"""The Rust backend, compiled into the package on install and used by default.

The Rust core in ``rust/`` implements LETOR parsing, feature binning, LambdaRank gradients,
histogram tree learning and NDCG. The numpy implementations in this package are the reference;
``tests/test_rust_parity.py`` checks agreement.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Literal

Backend = Literal["python", "rust"]
BACKENDS: tuple[str, ...] = ("auto", "python", "rust")


def _load_extension() -> Any:
    try:
        return importlib.import_module("learning_to_rank.learning_to_rank_rs")
    except ImportError:
        if os.environ.get("LTR_REQUIRE_RUST") == "1":
            raise
        return None


_EXTENSION: Any = _load_extension()


def rust_available() -> bool:
    """Return whether the Rust extension is importable."""

    return _EXTENSION is not None


def resolve_backend(backend: str) -> Backend:
    """Map ``auto`` to the fastest available backend and validate explicit choices."""

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "auto":
        return "rust" if rust_available() else "python"
    if backend == "rust" and not rust_available():
        raise RuntimeError(
            "the Rust backend is not built; run `uv sync` or `make rust-develop`, "
            "or use backend='python'"
        )
    return "rust" if backend == "rust" else "python"


def extension() -> Any:
    """Return the loaded extension module (only valid when ``rust_available()``)."""

    if _EXTENSION is None:
        raise RuntimeError("the Rust backend is not built")
    return _EXTENSION
