"""Vectorized ranking metrics: NDCG@k, MAP, MRR, ERR@k and precision@k, per query and averaged.

Conventions (stated because published numbers differ on them):

* Documents are ranked by descending score; equal scores keep their stored order.
* NDCG uses gain ``2^label - 1`` and discount ``1 / log2(rank + 1)`` (Burges et al., 2005), and
  the ideal DCG is computed at the same cutoff. A query with no relevant document gets NDCG 1.0,
  as in LightGBM and XGBoost; pass ``empty_ndcg=0.0`` for the other common convention.
* MAP and MRR treat ``label >= relevance_threshold`` (default 1) as relevant; a query without a
  relevant document scores 0. Average precision divides by all relevant documents of the query,
  as trec_eval does.
* ERR@k (Chapelle et al., 2009) uses stop probability ``(2^label - 1) / 2^max_grade``; MSLR and
  Istella grade 0 to 4, so ``max_grade = 4``.

Queries are processed in blocks of a padded ``queries x max_length`` matrix, so memory stays
bounded for long lists.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.data import RankingData

Array = NDArray[np.float64]
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
_BLOCK_CELLS = 20_000_000


def _check(
    y: NDArray[np.floating], scores: NDArray[np.floating], offsets: NDArray[np.integer]
) -> None:
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of equal length")
    if offsets.ndim != 1 or offsets.shape[0] < 1 or offsets[0] != 0 or offsets[-1] != y.shape[0]:
        raise ValueError("offsets must start at 0 and end at the number of documents")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be non-decreasing")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")


def _blocks(offsets: NDArray[np.int64]) -> Iterator[tuple[int, int]]:
    """Query ranges [a, b) whose padded matrix fits in the block budget."""

    sizes = np.diff(offsets)
    nq = sizes.shape[0]
    a = 0
    while a < nq:
        b = a + 1
        width = int(sizes[a])
        while b < nq and max(width, int(sizes[b])) * (b + 1 - a) <= _BLOCK_CELLS:
            width = max(width, int(sizes[b]))
            b += 1
        yield a, b
        a = b


def ranked_matrix(
    y: NDArray[np.floating], scores: NDArray[np.floating], offsets: NDArray[np.integer]
) -> Array:
    """Labels in ranked order, one row per query, padded with 0 to the longest list."""

    off = np.asarray(offsets, dtype=np.int64)
    sizes = np.diff(off)
    nq = sizes.shape[0]
    width = int(sizes.max()) if nq else 0
    out = np.zeros((nq, width), dtype=np.float64)
    if nq == 0 or width == 0:
        return out
    qi = np.repeat(np.arange(nq), sizes)
    by_score = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    order = by_score[np.argsort(qi[by_score], kind="stable")]
    pos = np.arange(off[-1] - off[0]) - (off[qi] - off[0])
    out[qi, pos] = np.asarray(y, dtype=np.float64)[order]
    return out


def _discounts(width: int) -> Array:
    return 1.0 / np.log2(np.arange(2, width + 2, dtype=np.float64))


def _dcg(labels: Array, k: int | None) -> Array:
    cut = labels if k is None else labels[:, :k]
    out: Array = ((np.exp2(cut) - 1.0) * _discounts(cut.shape[1])).sum(axis=1)
    return out


def _block_metrics(
    L: Array,
    ideal: Array,
    ks: Sequence[int],
    relevance_threshold: float,
    max_grade: float,
    empty_ndcg: float,
) -> dict[str, Array]:
    out: dict[str, Array] = {}
    width = L.shape[1]
    for k in ks:
        dcg, idcg = _dcg(L, k), _dcg(ideal, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"ndcg@{k}"] = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), empty_ndcg)
    dcg, idcg = _dcg(L, None), _dcg(ideal, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ndcg"] = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), empty_ndcg)
    rel = relevance_threshold <= L
    n_rel = rel.sum(axis=1)
    ranks = np.arange(1, width + 1, dtype=np.float64)
    hits = np.cumsum(rel, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["map"] = np.where(
            n_rel > 0, (hits / ranks * rel).sum(axis=1) / np.maximum(n_rel, 1), 0.0
        )
    first = np.argmax(rel, axis=1)
    out["mrr"] = np.where(n_rel > 0, 1.0 / (first + 1.0), 0.0)
    stop = (np.exp2(L) - 1.0) / np.exp2(max_grade)
    if np.any(stop > 1.0):
        raise ValueError(f"labels exceed max_grade={max_grade}")
    reach = np.cumprod(np.c_[np.ones(L.shape[0]), 1.0 - stop[:, :-1]], axis=1)
    contrib = stop * reach / ranks
    for k in ks:
        out[f"err@{k}"] = contrib[:, :k].sum(axis=1)
        out[f"precision@{k}"] = rel[:, :k].sum(axis=1) / float(k)
    return out


def per_query(
    y: NDArray[np.floating],
    scores: NDArray[np.floating],
    offsets: NDArray[np.integer],
    *,
    ks: Iterable[int] = DEFAULT_KS,
    relevance_threshold: float = 1.0,
    max_grade: float = 4.0,
    empty_ndcg: float = 1.0,
) -> dict[str, Array]:
    """Every metric for every query: ``ndcg@k``, ``ndcg``, ``map``, ``mrr``, ``err@k``,
    ``precision@k``."""

    y_arr = np.asarray(y, dtype=np.float64)
    s_arr = np.asarray(scores, dtype=np.float64)
    off = np.asarray(offsets, dtype=np.int64)
    _check(y_arr, s_arr, off)
    k_list = sorted({int(k) for k in ks})
    if any(k < 1 for k in k_list):
        raise ValueError("cutoffs must be positive")
    parts: list[dict[str, Array]] = []
    for a, b in _blocks(off):
        lo, hi = int(off[a]), int(off[b])
        sub = off[a : b + 1] - lo
        L = ranked_matrix(y_arr[lo:hi], s_arr[lo:hi], sub)
        ideal = ranked_matrix(y_arr[lo:hi], y_arr[lo:hi], sub)
        parts.append(_block_metrics(L, ideal, k_list, relevance_threshold, max_grade, empty_ndcg))
    if not parts:
        empty = _block_metrics(
            np.zeros((0, 1)), np.zeros((0, 1)), k_list, relevance_threshold, max_grade, empty_ndcg
        )
        return empty
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def evaluate(
    data: RankingData,
    scores: NDArray[np.floating],
    *,
    ks: Iterable[int] = DEFAULT_KS,
    relevance_threshold: float = 1.0,
    max_grade: float = 4.0,
    empty_ndcg: float = 1.0,
) -> dict[str, float]:
    """Mean of every per-query metric over the queries of ``data``."""

    pq = per_query(
        data.y,
        scores,
        data.offsets,
        ks=ks,
        relevance_threshold=relevance_threshold,
        max_grade=max_grade,
        empty_ndcg=empty_ndcg,
    )
    return {name: float(v.mean()) if v.size else float("nan") for name, v in pq.items()}


def ndcg_at(data: RankingData, scores: NDArray[np.floating], k: int = 10) -> float:
    """Mean NDCG@k (LightGBM convention for queries without relevant documents)."""

    return evaluate(data, scores, ks=(k,))[f"ndcg@{k}"]


# Readable single-query definitions, used as the reference in tests.


def ndcg_reference(labels: Sequence[float], scores: Sequence[float], k: int) -> float:
    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    dcg = sum((2.0 ** labels[i] - 1) / np.log2(r + 2) for r, i in enumerate(order[:k]))
    ideal = sorted(labels, reverse=True)
    idcg = sum((2.0**g - 1) / np.log2(r + 2) for r, g in enumerate(ideal[:k]))
    return float(dcg / idcg) if idcg > 0 else 1.0


def err_reference(
    labels: Sequence[float], scores: Sequence[float], k: int, max_grade: float = 4
) -> float:
    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    p_continue, total = 1.0, 0.0
    for r, i in enumerate(order[:k], start=1):
        stop = (2.0 ** labels[i] - 1) / 2.0**max_grade
        total += p_continue * stop / r
        p_continue *= 1 - stop
    return total
