"""Segment analysis: how ranking quality differs across groups of queries.

Search traffic is not uniform: a few head queries are issued very often and a long tail is issued
rarely, and short and long queries behave differently. Averages over all queries hide this, so
:func:`segment_report` breaks a metric down by segment for several systems and tests each
difference against a baseline with the paired randomization test, adjusting the p-values across
segments with Holm-Bonferroni.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.significance import holm, randomization_test


def frequency_segments(
    frequency: NDArray[np.floating],
    *,
    shares: Sequence[float] = (0.1, 0.4, 0.5),
    names: Sequence[str] = ("head", "torso", "tail"),
) -> NDArray[np.str_]:
    """Label queries by popularity: the most frequent ``shares[0]`` are ``names[0]``, and so on.

    Queries are ranked by descending frequency with ties broken by position, so the segment sizes
    match the shares exactly (up to rounding).
    """

    f = np.asarray(frequency, dtype=np.float64)
    if len(shares) != len(names) or not np.isclose(sum(shares), 1.0) or min(shares) <= 0:
        raise ValueError("shares must be positive, sum to 1 and match names")
    order = np.lexsort((np.arange(f.shape[0]), -f))
    bounds = np.round(np.cumsum(shares) * f.shape[0]).astype(int)
    out = np.empty(f.shape[0], dtype=object)
    start = 0
    for name, end in zip(names, bounds, strict=True):
        out[order[start:end]] = name
        start = end
    return out.astype(str)


def length_segments(lengths: NDArray[np.integer], *, max_exact: int = 4) -> NDArray[np.str_]:
    """``"1"`` .. ``"max_exact"`` terms, ``"<max_exact+1>+"`` above, ``"unknown"`` for 0."""

    n = np.asarray(lengths, dtype=np.int64)
    out = np.where(n > max_exact, f"{max_exact + 1}+", n.astype(str))
    return np.where(n <= 0, "unknown", out).astype(str)


@dataclass(frozen=True)
class SegmentRow:
    segment: str
    n_queries: int
    means: dict[str, float]
    diffs: dict[str, float]
    p_values: dict[str, float]
    adjusted_p_values: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def segment_report(
    per_query: Mapping[str, NDArray[np.floating]],
    segments: NDArray[np.str_],
    *,
    baseline: str,
    order: Sequence[str] | None = None,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> list[SegmentRow]:
    """Mean metric per segment for every system, with differences to ``baseline`` and
    randomization-test p-values (Holm-adjusted per system across segments)."""

    if baseline not in per_query:
        raise ValueError(f"baseline {baseline!r} is not one of the systems")
    seg = np.asarray(segments).astype(str)
    sizes = {name: np.asarray(v).shape for name, v in per_query.items()}
    if any(s != seg.shape for s in sizes.values()):
        raise ValueError("every system needs one value per query, aligned with segments")
    names = list(order) if order is not None else list(dict.fromkeys(seg.tolist()))
    base = np.asarray(per_query[baseline], dtype=np.float64)
    others = [s for s in per_query if s != baseline]
    raw: dict[str, list[float]] = {s: [] for s in others}
    rows: list[SegmentRow] = []
    for name in names:
        sel = seg == name
        means = {
            s: float(np.asarray(v, dtype=np.float64)[sel].mean()) if sel.any() else float("nan")
            for s, v in per_query.items()
        }
        diffs, pvals = {}, {}
        for s in others:
            x = np.asarray(per_query[s], dtype=np.float64)[sel]
            diffs[s] = float((x - base[sel]).mean()) if sel.any() else float("nan")
            p = (
                randomization_test(x, base[sel], n_permutations=n_permutations, seed=seed)
                if sel.sum() >= 2
                else float("nan")
            )
            pvals[s] = p
            raw[s].append(p)
        rows.append(SegmentRow(name, int(sel.sum()), means, diffs, pvals, {}))
    adjusted = {s: holm(np.nan_to_num(np.asarray(v), nan=1.0)) for s, v in raw.items()}
    return [
        SegmentRow(
            r.segment,
            r.n_queries,
            r.means,
            r.diffs,
            r.p_values,
            {s: float(adjusted[s][i]) for s in others},
        )
        for i, r in enumerate(rows)
    ]
