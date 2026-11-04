"""Query-grouped ranking data and LETOR (SVMlight) input and output.

A :class:`RankingData` holds a dense float32 feature matrix, one relevance label per row and one
query id per row, with the rows of each query stored contiguously. ``offsets`` delimits the
queries: query ``i`` is rows ``offsets[i]:offsets[i + 1]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.backend import extension, resolve_backend

FloatArray = NDArray[np.float32]


def group_offsets(qid: NDArray[np.int64]) -> NDArray[np.int64]:
    """Offsets of the contiguous runs of equal query ids; raises if a query id repeats later."""

    n = qid.shape[0]
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, qid[1:] != qid[:-1]])
    if np.unique(qid[starts]).shape[0] != starts.shape[0]:
        raise ValueError("rows of each query must be contiguous (a query id appears in two runs)")
    out: NDArray[np.int64] = np.r_[starts, n].astype(np.int64)
    return out


@dataclass(frozen=True)
class RankingData:
    """Dense features ``X`` (n x d, float32), labels ``y`` (n, float32), query ids ``qid`` (n)."""

    X: FloatArray
    y: FloatArray
    qid: NDArray[np.int64]
    offsets: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        X = np.ascontiguousarray(self.X, dtype=np.float32)
        y = np.ascontiguousarray(self.y, dtype=np.float32)
        qid = np.ascontiguousarray(self.qid, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.shape != (X.shape[0],) or qid.shape != (X.shape[0],):
            raise ValueError("X, y and qid must have the same number of rows")
        if not np.isfinite(y).all():
            raise ValueError("labels must be finite")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "offsets", group_offsets(qid))

    @classmethod
    def from_arrays(
        cls,
        X: NDArray[np.floating] | Sequence[Sequence[float]],
        y: NDArray[np.floating] | Sequence[float],
        qid: NDArray[np.integer] | Sequence[int],
        *,
        sort: bool = False,
    ) -> RankingData:
        """Build from arrays; ``sort=True`` groups rows by query id (stable within a query)."""

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        q_arr = np.asarray(qid, dtype=np.int64)
        if sort:
            order = np.argsort(q_arr, kind="stable")
            X_arr, y_arr, q_arr = X_arr[order], y_arr[order], q_arr[order]
        return cls(X_arr, y_arr, q_arr)

    @property
    def n_docs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.offsets.shape[0] - 1)

    @property
    def query_ids(self) -> NDArray[np.int64]:
        """The query id of each query, in storage order."""

        return self.qid[self.offsets[:-1]]

    @property
    def group_sizes(self) -> NDArray[np.int64]:
        return np.diff(self.offsets)

    def query_index(self) -> NDArray[np.int64]:
        """For each row, the position of its query (0 .. n_queries - 1)."""

        return np.repeat(np.arange(self.n_queries, dtype=np.int64), self.group_sizes)

    def subset(self, queries: NDArray[np.integer] | Sequence[int]) -> RankingData:
        """The queries at the given positions (not query ids), in the given order."""

        q = np.asarray(queries, dtype=np.int64)
        if q.size and (int(np.min(q)) < 0 or int(np.max(q)) >= self.n_queries):
            raise IndexError("query position out of range")
        starts, ends = self.offsets[q], self.offsets[q + 1]
        lengths = ends - starts
        rows = np.repeat(ends - np.cumsum(lengths), lengths) + np.arange(int(lengths.sum()))
        return RankingData(self.X[rows], self.y[rows], self.qid[rows])

    def with_labels(self, y: NDArray[np.floating]) -> RankingData:
        """The same rows with replacement labels (for example click-derived targets)."""

        return RankingData(self.X, np.asarray(y, dtype=np.float32), self.qid)

    @staticmethod
    def concat(parts: Sequence[RankingData]) -> RankingData:
        if not parts:
            raise ValueError("nothing to concatenate")
        return RankingData(
            np.concatenate([p.X for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.qid for p in parts]),
        )


def parse_svmlight_python(text: str, n_features: int = 0) -> RankingData:
    """Reference LETOR parser: ``<label> qid:<id> <index>:<value> ... [# comment]``."""

    labels: list[float] = []
    qids: list[int] = []
    rows: list[list[tuple[int, float]]] = []
    max_index = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].split()
        if not content:
            continue
        try:
            label = float(content[0])
        except ValueError as exc:
            raise ValueError(f"line {line_no}: invalid label {content[0]!r}") from exc
        if len(content) < 2 or not content[1].startswith("qid:"):
            raise ValueError(f"line {line_no}: missing qid")
        try:
            qid = int(content[1][4:])
        except ValueError as exc:
            raise ValueError(f"line {line_no}: invalid qid {content[1]!r}") from exc
        entries: list[tuple[int, float]] = []
        for token in content[2:]:
            index_text, sep, value_text = token.partition(":")
            try:
                index = int(index_text)
                value = float(value_text) if sep else float("nan")
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid feature {token!r}") from exc
            if not sep:
                raise ValueError(f"line {line_no}: invalid feature {token!r}")
            if index < 1:
                raise ValueError(f"line {line_no}: feature indices start at 1")
            max_index = max(max_index, index)
            entries.append((index - 1, value))
        labels.append(label)
        qids.append(qid)
        rows.append(entries)
    if n_features and max_index > n_features:
        raise ValueError(f"feature index {max_index} exceeds n_features={n_features}")
    d = n_features or max_index
    X = np.zeros((len(rows), d), dtype=np.float32)
    for i, entries in enumerate(rows):
        for j, v in entries:
            X[i, j] = v
    return RankingData(X, np.asarray(labels, dtype=np.float32), np.asarray(qids, dtype=np.int64))


def parse_svmlight(text: str, n_features: int = 0, backend: str = "auto") -> RankingData:
    """Parse LETOR text with the Rust parser (default) or the Python reference."""

    if resolve_backend(backend) == "python":
        return parse_svmlight_python(text, n_features)
    labels, qids, feats, d = extension().parse_svmlight(text, n_features)
    y = np.frombuffer(labels, dtype=np.float32)
    q = np.frombuffer(qids, dtype=np.int64)
    X = np.frombuffer(feats, dtype=np.float32).reshape(y.shape[0], d)
    return RankingData(X, y, q)


def read_svmlight(path: str | Path, n_features: int = 0, backend: str = "auto") -> RankingData:
    """Read a LETOR file (plain text or ``.gz``)."""

    p = Path(path)
    if p.suffix == ".gz":
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = p.read_text(encoding="utf-8")
    return parse_svmlight(text, n_features, backend)


def to_svmlight(data: RankingData, *, precision: int = 6) -> str:
    """Serialize to LETOR text (zero features are written too, so the width round-trips)."""

    lines = []
    for i in range(data.n_docs):
        feats = " ".join(f"{j + 1}:{v:.{precision}g}" for j, v in enumerate(data.X[i]))
        label = f"{data.y[i]:g}"
        lines.append(f"{label} qid:{data.qid[i]} {feats}")
    return "\n".join(lines) + "\n"


def synthetic(
    n_queries: int = 200,
    docs_per_query: tuple[int, int] = (10, 60),
    n_features: int = 20,
    *,
    noise: float = 0.5,
    seed: int = 0,
) -> RankingData:
    """A synthetic graded-relevance dataset with a nonlinear relevance function.

    Relevance (0..4) is a thresholded mix of feature interactions plus Gaussian noise, so trees
    beat a linear model and the label distribution is skewed towards 0 as in web search data.
    """

    rng = np.random.default_rng(seed)
    sizes = rng.integers(docs_per_query[0], docs_per_query[1] + 1, size=n_queries)
    n = int(sizes.sum())
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    query_shift = np.repeat(rng.normal(scale=0.5, size=n_queries), sizes)
    k = min(n_features, 5)
    signal = (
        1.2 * X[:, 0]
        + 0.8 * np.tanh(X[:, 1 % k] * X[:, 2 % k])
        + 0.6 * (X[:, 3 % k] > 0.5)
        - 0.4 * X[:, 4 % k] ** 2
        + query_shift
        + rng.normal(scale=noise, size=n)
    )
    thresholds = np.quantile(signal, [0.55, 0.8, 0.92, 0.98])
    y = np.searchsorted(thresholds, signal).astype(np.float32)
    qid = np.repeat(np.arange(n_queries, dtype=np.int64), sizes)
    return RankingData(X, y, qid)
