from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_to_rank.data import RankingData, synthetic
from learning_to_rank.metrics import (
    err_reference,
    evaluate,
    ndcg_at,
    ndcg_reference,
    per_query,
    ranked_matrix,
)


def one(labels: list[float], scores: list[float], **kw: object) -> dict[str, float]:
    off = np.array([0, len(labels)])
    pq = per_query(np.array(labels), np.array(scores), off, **kw)  # type: ignore[arg-type]
    return {k: float(v[0]) for k, v in pq.items()}


def test_hand_computed_values() -> None:
    m = one([3, 2, 0, 1], [0.1, 0.4, 0.3, 0.2], ks=(1, 2, 4))
    # Ranking by score: labels 2, 0, 1, 3.
    dcg4 = 3 / np.log2(2) + 0 + 1 / np.log2(4) + 7 / np.log2(5)
    idcg4 = 7 / np.log2(2) + 3 / np.log2(3) + 1 / np.log2(4)
    assert m["ndcg@4"] == pytest.approx(dcg4 / idcg4)
    assert m["ndcg@1"] == pytest.approx(3 / 7)
    # Relevant: positions 1, 3, 4 -> AP = (1/1 + 2/3 + 3/4) / 3.
    assert m["map"] == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert m["mrr"] == 1.0
    assert m["precision@2"] == 0.5
    r = [3 / 16, 0, 1 / 16, 7 / 16]
    err = r[0] + (1 - r[0]) * r[1] / 2 + (1 - r[0]) * (1 - r[1]) * r[2] / 3
    err += (1 - r[0]) * (1 - r[1]) * (1 - r[2]) * r[3] / 4
    assert m["err@4"] == pytest.approx(err)


def test_conventions_for_empty_queries() -> None:
    m = one([0, 0], [1, 2])
    assert m["ndcg@10"] == 1.0 and m["map"] == 0.0 and m["mrr"] == 0.0 and m["err@10"] == 0.0
    assert one([0, 0], [1, 2], empty_ndcg=0.0)["ndcg@10"] == 0.0


def test_ties_keep_document_order() -> None:
    assert one([0, 1], [5, 5], ks=(1,))["ndcg@1"] == 0.0
    assert one([1, 0], [5, 5], ks=(1,))["ndcg@1"] == 1.0
    L = ranked_matrix(np.array([0.0, 1, 2, 3]), np.array([1.0, 1, 0, 2]), np.array([0, 2, 4]))
    assert L.tolist() == [[0.0, 1.0], [3.0, 2.0]]


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        per_query(np.zeros(2), np.zeros(3), np.array([0, 2]))
    with pytest.raises(ValueError):
        per_query(np.zeros(2), np.zeros(2), np.array([0, 3]))
    with pytest.raises(ValueError):
        per_query(np.zeros(2), np.array([0.0, np.nan]), np.array([0, 2]))
    with pytest.raises(ValueError):
        per_query(np.zeros(2), np.zeros(2), np.array([0, 2]), ks=(0,))
    with pytest.raises(ValueError):
        per_query(np.array([5.0]), np.zeros(1), np.array([0, 1]))
    with pytest.raises(ValueError):
        per_query(np.zeros(3), np.zeros(3), np.array([0, 2, 1, 3]))


def test_empty_dataset() -> None:
    pq = per_query(np.zeros(0), np.zeros(0), np.array([0]))
    assert pq["ndcg@10"].shape == (0,)
    d = RankingData.from_arrays(np.zeros((0, 1)), [], [])
    assert np.isnan(evaluate(d, np.zeros(0))["map"])


def test_blocks_cover_long_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    import learning_to_rank.metrics as metrics

    data = synthetic(50, (3, 40), 4, seed=5)
    s = np.random.default_rng(0).normal(size=data.n_docs)
    full = evaluate(data, s)
    monkeypatch.setattr(metrics, "_BLOCK_CELLS", 50)
    assert evaluate(data, s) == pytest.approx(full)


@settings(max_examples=60, deadline=None)
@given(
    labels=st.lists(st.integers(0, 4), min_size=1, max_size=25),
    seed=st.integers(0, 10_000),
    k=st.integers(1, 12),
)
def test_against_reference_definitions(labels: list[int], seed: int, k: int) -> None:
    scores = np.random.default_rng(seed).permutation(len(labels)).astype(float)
    m = one([float(x) for x in labels], scores.tolist(), ks=(k,))
    assert m[f"ndcg@{k}"] == pytest.approx(ndcg_reference(labels, scores.tolist(), k))
    assert m[f"err@{k}"] == pytest.approx(err_reference(labels, scores.tolist(), k))
    assert 0.0 <= m[f"ndcg@{k}"] <= 1.0 + 1e-12
    assert 0.0 <= m["map"] <= 1.0 and 0.0 <= m["mrr"] <= 1.0


@settings(max_examples=40, deadline=None)
@given(labels=st.lists(st.integers(0, 4), min_size=1, max_size=30))
def test_ideal_ranking_is_perfect(labels: list[int]) -> None:
    y = np.array(labels, dtype=float)
    m = one(y.tolist(), (y + np.linspace(0, 0.1, len(y))).tolist())
    assert m["ndcg"] == pytest.approx(1.0)
    if y.max() >= 1:
        assert m["mrr"] == 1.0


def test_monotone_score_transform_is_invariant() -> None:
    data = synthetic(40, seed=6)
    s = np.random.default_rng(1).normal(size=data.n_docs)
    assert evaluate(data, s) == pytest.approx(evaluate(data, np.exp(s) * 3 + 1))
    assert ndcg_at(data, s, 5) == pytest.approx(evaluate(data, s)["ndcg@5"])


def test_sklearn_ndcg_agrees() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    data = synthetic(30, (5, 20), 4, seed=8)
    s = np.random.default_rng(2).normal(size=data.n_docs)
    ours = per_query(data.y, s, data.offsets, ks=(5,))["ndcg@5"]
    for q in range(data.n_queries):
        a, b = data.offsets[q], data.offsets[q + 1]
        gains = np.exp2(data.y[a:b]) - 1
        if gains.sum() == 0:
            continue
        ref = sklearn_metrics.ndcg_score([gains], [s[a:b]], k=5)
        assert ours[q] == pytest.approx(ref)
