"""The Rust core against the numpy reference: gradients, trees and NDCG."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_to_rank import LambdaMART
from learning_to_rank.backend import extension, resolve_backend, rust_available
from learning_to_rank.data import RankingData, synthetic
from learning_to_rank.gbdt_reference import lambdarank_gradients_reference
from learning_to_rank.metrics import evaluate

pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


@settings(max_examples=60, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 40), min_size=1, max_size=8),
    seed=st.integers(0, 1000),
    trunc=st.integers(1, 40),
    sigma=st.floats(0.5, 2.0),
    norm=st.booleans(),
    ties=st.booleans(),
)
def test_gradients_match_reference(
    sizes: list[int], seed: int, trunc: int, sigma: float, norm: bool, ties: bool
) -> None:
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    y = rng.integers(0, 5, size=n).astype(np.float32)
    s = rng.normal(size=n)
    if ties:
        s = np.round(s)
    off = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    g_ref, h_ref = lambdarank_gradients_reference(y.astype(np.float64), s, off, trunc, sigma, norm)
    g_raw, h_raw = extension().lambdarank_gradients(y, s, off, trunc, sigma, norm)
    np.testing.assert_allclose(np.frombuffer(g_raw), g_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.frombuffer(h_raw), h_ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("feature_fraction", [1.0, 0.6])
@pytest.mark.parametrize("objective", ["lambdarank", "regression"])
def test_trees_match_reference(feature_fraction: float, objective: str) -> None:
    train = synthetic(40, (5, 25), 6, seed=21)
    valid = synthetic(15, (5, 25), 6, seed=22)
    kw = {
        "n_estimators": 6,
        "num_leaves": 9,
        "min_data_in_leaf": 5,
        "feature_fraction": feature_fraction,
        "objective": objective,
        "max_depth": 4,
        "lambda_l2": 0.5,
        "seed": 3,
    }
    r = LambdaMART(backend="rust", **kw).fit(train, valid)  # type: ignore[arg-type]
    p = LambdaMART(backend="python", **kw).fit(train, valid)  # type: ignore[arg-type]
    assert r.ensemble is not None and p.ensemble is not None
    np.testing.assert_array_equal(r.ensemble.split_feature, p.ensemble.split_feature)
    np.testing.assert_array_equal(r.ensemble.threshold, p.ensemble.threshold)
    np.testing.assert_array_equal(r.ensemble.left, p.ensemble.left)
    np.testing.assert_allclose(r.ensemble.leaf_value, p.ensemble.leaf_value, rtol=1e-10)
    np.testing.assert_allclose(r.result.valid_ndcg, p.result.valid_ndcg, rtol=1e-12)
    assert r.result.train_ndcg == pytest.approx(p.result.train_ndcg, rel=1e-12)


def test_large_leaves_use_chunked_histograms() -> None:
    """More than 65,536 rows per leaf exercises the parallel chunked histogram path."""

    train = synthetic(1000, (70, 100), 5, seed=31)
    assert train.n_docs > 65_536
    kw = {"n_estimators": 2, "num_leaves": 5}
    r = LambdaMART(backend="rust", **kw).fit(train)  # type: ignore[arg-type]
    p = LambdaMART(backend="python", **kw).fit(train)  # type: ignore[arg-type]
    assert r.ensemble is not None and p.ensemble is not None
    np.testing.assert_array_equal(r.ensemble.split_feature, p.ensemble.split_feature)
    np.testing.assert_allclose(r.ensemble.leaf_value, p.ensemble.leaf_value, rtol=1e-10)


def test_mean_ndcg_matches_metrics_module() -> None:
    data = synthetic(50, seed=9)
    s = np.random.default_rng(4).normal(size=data.n_docs)
    s[::3] = 0.0
    for k in (1, 5, 10):
        rust = extension().mean_ndcg(data.y, s, data.offsets, k)
        assert rust == pytest.approx(evaluate(data, s, ks=(k,))[f"ndcg@{k}"], rel=1e-12)


def test_extension_rejects_bad_inputs() -> None:
    ext = extension()
    y = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        ext.lambdarank_gradients(y, np.zeros(3), np.array([0, 2], dtype=np.int64))
    with pytest.raises(ValueError):
        ext.mean_ndcg(y, np.zeros(2), np.array([0, 3], dtype=np.int64), 3)
    with pytest.raises(ValueError):
        ext.lambdarank_gradients(y, np.zeros(3), np.array([0, -1, 3], dtype=np.int64))
    X = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        ext.fit_bin_edges(X, 2, np.array([5], dtype=np.uint32), 16)
    with pytest.raises(ValueError):
        ext.fit_bin_edges(X, 2, np.array([0], dtype=np.uint32), 1)
    with pytest.raises(ValueError):
        ext.fit_bin_edges(X, 3, np.array([0], dtype=np.uint32), 16)
    with pytest.raises(ValueError):
        ext.apply_bins(X, 2, [[0.0]], np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        fortran = np.asfortranarray(np.ones((2, 3), dtype=np.float32))
        ext.apply_bins(fortran, 3, [[], [], []], np.zeros((2, 3), dtype=np.uint8))
    bins = np.zeros((3, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        ext.Booster(bins, 1, y, np.array([0, 3], dtype=np.int64), [[]], objective="pairwise")
    with pytest.raises(ValueError):
        ext.Booster(bins, 1, y, np.array([0, 3], dtype=np.int64), [[]], valid_bins=bins)
    with pytest.raises(ValueError):
        ext.Booster(bins, 1, y, np.array([0, 2], dtype=np.int64), [[]])
    with pytest.raises(ValueError):
        ext.predict(
            X,
            2,
            *[
                np.zeros(1, dtype=t)
                for t in (np.int64, np.int32, np.float32, np.int32, np.int32, np.int64, np.float64)
            ],
        )


def test_booster_timings_and_scores() -> None:
    data = synthetic(30, seed=2)
    m = LambdaMART(n_estimators=3).fit(data)
    assert m.ensemble is not None
    from learning_to_rank.binning import BinMapper

    mapper = BinMapper.fit(data.X)
    b = extension().Booster(
        mapper.transform(data.X), data.n_features, data.y, data.offsets, mapper.to_lists()
    )
    b.step(2)
    names = [k for k, _ in b.timings()]
    assert names[:2] == ["gradients", "histograms"]
    assert np.frombuffer(b.train_scores()).shape == (data.n_docs,)
    assert b.n_trees == 2 and extension().num_threads() >= 1


def test_backend_resolution() -> None:
    assert resolve_backend("auto") == "rust"
    with pytest.raises(ValueError):
        resolve_backend("cuda")


def test_ranking_data_is_contiguous_for_extension() -> None:
    d = synthetic(5, seed=1)
    sliced = RankingData(d.X[:, ::-1], d.y, d.qid)
    assert sliced.X.flags["C_CONTIGUOUS"]
    LambdaMART(n_estimators=1).fit(sliced)
