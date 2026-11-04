"""Feature binning for the histogram tree learner.

Edges are fitted per feature on a random sample of rows (200,000 by default, as LightGBM's
``bin_construct_sample_cnt``). Each edge is an observed value: bin ``b`` holds values in
``(edges[b-1], edges[b]]``, so a split "bin <= t" is the raw-feature test "x <= edges[t]" and the
model predicts on raw features without binning them. NaN goes to bin 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.backend import extension, resolve_backend


def fit_edges_reference(values: NDArray[np.floating], max_bin: int) -> NDArray[np.float32]:
    """Greedy equal-frequency edges over distinct values (mirrors ``rust/src/binning.rs``)."""

    if max_bin < 2:
        raise ValueError("max_bin must be at least 2")
    v = np.asarray(values, dtype=np.float32)
    v = v[~np.isnan(v)]
    distinct, counts = np.unique(v, return_counts=True)
    if distinct.shape[0] <= 1:
        return np.zeros(0, dtype=np.float32)
    if distinct.shape[0] <= max_bin:
        return distinct[:-1].astype(np.float32)
    edges: list[float] = []
    remaining_count = int(v.shape[0])
    remaining_bins = max_bin
    in_bin = 0
    last = distinct.shape[0] - 1
    for i in range(distinct.shape[0]):
        if i == last or remaining_bins == 1:
            break
        in_bin += int(counts[i])
        if in_bin * remaining_bins >= remaining_count:
            edges.append(float(distinct[i]))
            remaining_count -= in_bin
            remaining_bins -= 1
            in_bin = 0
    return np.asarray(edges, dtype=np.float32)


@dataclass
class BinMapper:
    """Per-feature bin edges; ``transform`` maps a float32 matrix to row-major uint8 bins."""

    edges: list[NDArray[np.float32]]

    @property
    def n_features(self) -> int:
        return len(self.edges)

    @property
    def n_bins(self) -> list[int]:
        return [int(e.shape[0]) + 1 for e in self.edges]

    @classmethod
    def fit(
        cls,
        X: NDArray[np.floating],
        *,
        max_bin: int = 255,
        sample_size: int = 200_000,
        seed: int = 0,
        backend: str = "auto",
    ) -> BinMapper:
        if not 2 <= max_bin <= 256:
            raise ValueError("max_bin must be in [2, 256]")
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        n, d = Xc.shape
        if n > sample_size:
            rng = np.random.default_rng(seed)
            sample = np.sort(rng.choice(n, size=sample_size, replace=False)).astype(np.uint32)
        else:
            sample = np.arange(n, dtype=np.uint32)
        if resolve_backend(backend) == "rust" and d > 0:
            raw = extension().fit_bin_edges(Xc, d, sample, max_bin)
            return cls([np.asarray(e, dtype=np.float32) for e in raw])
        rows = Xc[sample]
        return cls([fit_edges_reference(rows[:, j], max_bin) for j in range(d)])

    def transform(self, X: NDArray[np.floating], backend: str = "auto") -> NDArray[np.uint8]:
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        if Xc.ndim != 2 or Xc.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got shape {Xc.shape}")
        out = np.empty(Xc.shape, dtype=np.uint8)
        if resolve_backend(backend) == "rust" and Xc.size:
            extension().apply_bins(Xc, self.n_features, [e.tolist() for e in self.edges], out)
            return out
        for j, e in enumerate(self.edges):
            col = Xc[:, j]
            b = np.searchsorted(e, col, side="left")
            b[np.isnan(col)] = 0
            out[:, j] = b
        return out

    def to_lists(self) -> list[list[float]]:
        return [e.tolist() for e in self.edges]

    @classmethod
    def from_lists(cls, edges: list[list[float]]) -> BinMapper:
        return cls([np.asarray(e, dtype=np.float32) for e in edges])
