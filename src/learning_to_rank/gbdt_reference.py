"""Numpy reference implementation of the LambdaMART trainer.

Readable, slow and written to perform the same floating-point operations in the same order as the
Rust core (sequential sums via ``cumsum``, the same histogram chunking, the same tie rules), so the
parity tests can require identical trees on small data. Use the Rust backend for real work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def _seq_sum(values: F64, start: float = 0.0) -> float:
    """Left-to-right sum (numpy's ``sum`` is pairwise, which rounds differently)."""

    if values.size == 0:
        return start
    return float(np.cumsum(np.r_[start, values])[-1])


def _gain(labels: F64) -> F64:
    return np.exp2(labels) - 1.0


def _discount(rank0: NDArray[np.int64] | int) -> F64:
    return 1.0 / np.log2(np.asarray(rank0, dtype=np.float64) + 2.0)


def ideal_dcg_reference(labels: NDArray[np.floating], k: int) -> float:
    ideal = np.sort(np.asarray(labels, dtype=np.float64))[::-1][:k]
    return _seq_sum(_gain(ideal) * _discount(np.arange(ideal.shape[0])))


def query_gradients_reference(
    labels: F64,
    scores: F64,
    inv_max_dcg: float,
    truncation_level: int = 30,
    sigma: float = 1.0,
    norm: bool = True,
) -> tuple[F64, F64]:
    """LambdaRank gradients of one query (see ``rust/src/lambda.rs`` for the definition)."""

    cnt = labels.shape[0]
    g = np.zeros(cnt)
    h = np.zeros(cnt)
    if cnt < 2 or inv_max_dcg <= 0:
        return g, h
    order = np.argsort(-scores, kind="stable")
    best, worst = scores[order[0]], scores[order[-1]]
    lab = labels.astype(np.float64)
    pair_sums: list[F64] = []
    for i in range(min(truncation_level, cnt - 1)):
        js = np.arange(i + 1, cnt)
        li, lj = lab[order[i]], lab[order[js]]
        keep = lj != li
        js = js[keep]
        if js.size == 0:
            continue
        i_high = li > lab[order[js]]
        high_rank = np.where(i_high, i, js)
        low_rank = np.where(i_high, js, i)
        high, low = order[high_rank], order[low_rank]
        delta_score = scores[high] - scores[low]
        dcg_gap = _gain(lab[high]) - _gain(lab[low])
        paired = np.abs(_discount(high_rank) - _discount(low_rank))
        delta = dcg_gap * paired * inv_max_dcg
        if norm and best != worst:
            delta = delta / (0.01 + np.abs(delta_score))
        p_lambda = 1.0 / (1.0 + np.exp(sigma * delta_score))
        p_hess = p_lambda * (1.0 - p_lambda)
        p_lambda = p_lambda * (-sigma * delta)
        p_hess = p_hess * (sigma * sigma * delta)
        # Document order[i] receives one contribution per j, in j order.
        di = order[i]
        g_i = np.where(i_high, p_lambda, -p_lambda)
        g[di] = _seq_sum(g_i, g[di])
        h[di] = _seq_sum(p_hess, h[di])
        # Every other document receives exactly one contribution from this i.
        dj = order[js]
        g[dj] = g[dj] + np.where(i_high, -p_lambda, p_lambda)
        h[dj] = h[dj] + p_hess
        pair_sums.append(-2.0 * p_lambda)
    total = _seq_sum(np.concatenate(pair_sums)) if pair_sums else 0.0
    if norm and total > 0:
        factor = np.log2(1.0 + total) / total
        g *= factor
        h *= factor
    return g, h


def lambdarank_gradients_reference(
    y: F64,
    scores: F64,
    offsets: NDArray[np.int64],
    truncation_level: int = 30,
    sigma: float = 1.0,
    norm: bool = True,
) -> tuple[F64, F64]:
    g = np.zeros(y.shape[0])
    h = np.zeros(y.shape[0])
    for q in range(offsets.shape[0] - 1):
        s, e = int(offsets[q]), int(offsets[q + 1])
        idcg = ideal_dcg_reference(y[s:e], truncation_level)
        inv = 1.0 / idcg if idcg > 0 else 0.0
        g[s:e], h[s:e] = query_gradients_reference(
            y[s:e].astype(np.float64), scores[s:e], inv, truncation_level, sigma, norm
        )
    return g, h


def n_chunks(length: int) -> int:
    return int(min(max(-(-length // 4096), 1), 16))


def build_histogram_reference(
    bins: NDArray[np.uint8], n_bins: list[int], rows: NDArray[np.int64], g: F64, h: F64
) -> list[tuple[F64, F64, NDArray[np.int64]]]:
    """Per feature: (sum of gradients, sum of hessians, count) per bin."""

    d = bins.shape[1]
    k = n_chunks(rows.shape[0])
    n = rows.shape[0]
    out: list[tuple[F64, F64, NDArray[np.int64]]] = []
    for j in range(d):
        tg = np.zeros(n_bins[j])
        th = np.zeros(n_bins[j])
        tc = np.zeros(n_bins[j], dtype=np.int64)
        for c in range(k):
            r = rows[c * n // k : (c + 1) * n // k]
            b = bins[r, j]
            tg = tg + np.bincount(b, weights=g[r], minlength=n_bins[j])
            th = th + np.bincount(b, weights=h[r], minlength=n_bins[j])
            tc = tc + np.bincount(b, minlength=n_bins[j])
        out.append((tg, th, tc))
    return out


@dataclass
class _Split:
    gain: float
    feature: int
    bin: int
    left_g: float
    left_h: float
    left_c: int


def _leaf_gain(g: F64 | float, h: F64 | float, lam: float) -> F64:
    den = np.asarray(h, dtype=np.float64) + lam
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.asarray(g) * np.asarray(g) / np.where(den > 0, den, 1.0), 0.0)


def best_split_reference(
    hist: list[tuple[F64, F64, NDArray[np.int64]]],
    mask: NDArray[np.bool_],
    sum_g: float,
    sum_h: float,
    count: int,
    p: dict[str, float],
) -> _Split | None:
    parent = float(_leaf_gain(sum_g, sum_h, p["lambda_l2"]))
    best: _Split | None = None
    for j, (hg, hh, hc) in enumerate(hist):
        if not mask[j] or hg.shape[0] < 2:
            continue
        gl = np.cumsum(hg[:-1])
        hl = np.cumsum(hh[:-1])
        cl = np.cumsum(hc[:-1])
        cr = count - cl
        gr = sum_g - gl
        hr = sum_h - hl
        valid = (
            (cl >= p["min_data_in_leaf"])
            & (hl >= p["min_sum_hessian_in_leaf"])
            & (cr >= p["min_data_in_leaf"])
            & (hr >= p["min_sum_hessian_in_leaf"])
        )
        gain = _leaf_gain(gl, hl, p["lambda_l2"]) + _leaf_gain(gr, hr, p["lambda_l2"]) - parent
        gain = np.where(valid & (gain > p["min_gain_to_split"]), gain, -np.inf)
        t = int(np.argmax(gain))
        if np.isfinite(gain[t]) and (best is None or gain[t] > best.gain):
            best = _Split(float(gain[t]), j, t, float(gl[t]), float(hl[t]), int(cl[t]))
    return best


@dataclass
class _Leaf:
    begin: int
    end: int
    g: float
    h: float
    c: int
    depth: int
    hist: list[tuple[F64, F64, NDArray[np.int64]]]
    split: _Split | None
    parent: int
    is_left: bool


@dataclass
class TreeArrays:
    split_feature: list[int] = field(default_factory=list)
    threshold_bin: list[int] = field(default_factory=list)
    threshold: list[float] = field(default_factory=list)
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    split_gain: list[float] = field(default_factory=list)
    leaf_value: list[float] = field(default_factory=list)
    leaf_count: list[int] = field(default_factory=list)


def grow_tree_reference(
    bins: NDArray[np.uint8],
    n_bins: list[int],
    edges: list[NDArray[np.float32]],
    g: F64,
    h: F64,
    indices: NDArray[np.int64],
    mask: NDArray[np.bool_],
    p: dict[str, float],
    shrinkage: float,
) -> tuple[TreeArrays, list[tuple[int, int]]]:
    """Leaf-wise growth with histogram subtraction (mirrors ``rust/src/tree.rs``)."""

    num_leaves = int(p["num_leaves"])
    max_depth = int(p["max_depth"])
    min_data = int(p["min_data_in_leaf"])

    def can_split(depth: int, c: int) -> bool:
        return (max_depth == 0 or depth < max_depth) and c >= 2 * max(min_data, 1)

    root_hist = build_histogram_reference(bins, n_bins, indices, g, h)
    sg, sh = _seq_sum(g[indices]), _seq_sum(h[indices])
    count = int(indices.shape[0])
    root_split = (
        best_split_reference(root_hist, mask, sg, sh, count, p)
        if can_split(0, count) and num_leaves > 1
        else None
    )
    leaves = [_Leaf(0, count, sg, sh, count, 0, root_hist, root_split, -1, True)]
    tree = TreeArrays()
    while len(leaves) < num_leaves:
        chosen = None
        for i, leaf in enumerate(leaves):
            if leaf.split is not None and (
                chosen is None or leaf.split.gain > leaves[chosen].split.gain  # type: ignore[union-attr]
            ):
                chosen = i
        if chosen is None:
            break
        leaf = leaves[chosen]
        s = leaf.split
        assert s is not None
        rows = indices[leaf.begin : leaf.end]
        go_left = bins[rows, s.feature] <= s.bin
        left_rows, right_rows = rows[go_left], rows[~go_left]
        mid = leaf.begin + left_rows.shape[0]
        indices[leaf.begin : mid] = left_rows
        indices[mid : leaf.end] = right_rows
        node = len(tree.split_feature)
        if leaf.parent >= 0:
            if leaf.is_left:
                tree.left[leaf.parent] = node
            else:
                tree.right[leaf.parent] = node
        new_leaf = len(leaves)
        tree.split_feature.append(s.feature)
        tree.threshold_bin.append(s.bin)
        tree.threshold.append(float(edges[s.feature][s.bin]))
        tree.left.append(~chosen)
        tree.right.append(~new_leaf)
        tree.split_gain.append(s.gain)
        depth = leaf.depth + 1
        lg, lh, lc = s.left_g, s.left_h, s.left_c
        rg, rh, rc = leaf.g - lg, leaf.h - lh, leaf.c - lc
        left_smaller = (mid - leaf.begin) <= (leaf.end - mid)
        small_rows = indices[leaf.begin : mid] if left_smaller else indices[mid : leaf.end]
        small = build_histogram_reference(bins, n_bins, small_rows, g, h)
        large = [
            (pg - sg_, ph - sh_, pc - sc)
            for (pg, ph, pc), (sg_, sh_, sc) in zip(leaf.hist, small, strict=True)
        ]
        left_hist, right_hist = (small, large) if left_smaller else (large, small)
        room = len(leaves) + 1 < num_leaves
        left_split = (
            best_split_reference(left_hist, mask, lg, lh, lc, p)
            if room and can_split(depth, lc)
            else None
        )
        right_split = (
            best_split_reference(right_hist, mask, rg, rh, rc, p)
            if room and can_split(depth, rc)
            else None
        )
        leaves[chosen] = _Leaf(
            leaf.begin, mid, lg, lh, lc, depth, left_hist, left_split, node, True
        )
        leaves.append(_Leaf(mid, leaf.end, rg, rh, rc, depth, right_hist, right_split, node, False))
    ranges = []
    lam = p["lambda_l2"]
    for leaf in leaves:
        den = leaf.h + lam
        out = -leaf.g / den if den > 0 else 0.0
        tree.leaf_value.append(out * shrinkage)
        tree.leaf_count.append(leaf.c)
        ranges.append((leaf.begin, leaf.end))
    return tree, ranges


class SplitMix64:
    """Same generator as ``rust/src/boost.rs`` (for identical feature subsampling)."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return self.next_u64() % bound


def feature_mask_reference(d: int, fraction: float, rng: SplitMix64) -> NDArray[np.bool_]:
    if fraction >= 1.0 or d == 0:
        return np.ones(d, dtype=bool)
    # Rust's `as usize` after round() is round-half-away-from-zero.
    k = min(max(int(np.floor(fraction * d + 0.5)), 1), d)
    perm = list(range(d))
    for i in range(k):
        j = i + rng.below(d - i)
        perm[i], perm[j] = perm[j], perm[i]
    mask = np.zeros(d, dtype=bool)
    mask[perm[:k]] = True
    return mask
