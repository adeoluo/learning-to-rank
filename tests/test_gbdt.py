from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import BACKENDS
from learning_to_rank import GBDTParams, LambdaMART, RankingData, evaluate
from learning_to_rank.data import synthetic
from learning_to_rank.gbdt import Ensemble


@pytest.mark.parametrize("backend", BACKENDS)
def test_learns_and_beats_random(backend: str, medium: tuple[RankingData, ...]) -> None:
    train, valid, test = medium
    model = LambdaMART(backend=backend, n_estimators=15, num_leaves=7).fit(train, valid)
    s = model.predict(test)
    ours = evaluate(test, s)["ndcg@10"]
    rand = evaluate(test, np.random.default_rng(0).normal(size=test.n_docs))["ndcg@10"]
    assert ours > rand + 0.2
    assert len(model.result.valid_ndcg) == 15
    assert model.result.fit_seconds > 0
    assert model.ensemble is not None and model.ensemble.n_trees == 15


def test_early_stopping_keeps_best(medium: tuple[RankingData, ...]) -> None:
    train, valid, _ = medium
    m = LambdaMART(n_estimators=300, early_stopping_rounds=5, learning_rate=0.5).fit(train, valid)
    hist = m.result.valid_ndcg
    assert m.result.best_iteration == int(np.argmax(hist)) + 1
    assert m.ensemble is not None and m.ensemble.n_trees == m.result.best_iteration
    assert len(hist) < 300
    assert np.isnan(m.result.train_ndcg)


def test_callback_and_no_validation(small: RankingData) -> None:
    seen: list[tuple[int, float | None]] = []
    m = LambdaMART(n_estimators=4, num_leaves=3, min_data_in_leaf=2).fit(
        small, callback=lambda i, v: seen.append((i, v)), chunk=3
    )
    assert [i for i, _ in seen] == [1, 2, 3, 4] and all(v is None for _, v in seen)
    assert m.result.valid_ndcg == [] and 0 < m.result.train_ndcg <= 1


def test_regression_objective(medium: tuple[RankingData, ...]) -> None:
    train, _, test = medium
    m = LambdaMART(objective="regression", n_estimators=30).fit(train)
    assert evaluate(test, m.predict(test.X))["ndcg@10"] > 0.6


def test_save_load_round_trip(tmp_path: Path, medium: tuple[RankingData, ...]) -> None:
    train, _, test = medium
    m = LambdaMART(n_estimators=5, feature_fraction=0.5).fit(train)
    path = tmp_path / "model.json"
    m.save(path)
    loaded = LambdaMART.load(path)
    np.testing.assert_array_equal(loaded.predict(test), m.predict(test))
    assert loaded.params == m.params
    gain = loaded.feature_importance()
    split = loaded.feature_importance("split")
    assert gain.shape == (train.n_features,) and split.sum() == loaded.ensemble.split_feature.size  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        loaded.feature_importance("cover")  # type: ignore[arg-type]
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        LambdaMART.load(bad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_backends_agree(backend: str, medium: tuple[RankingData, ...]) -> None:
    train, _, test = medium
    m = LambdaMART(n_estimators=8).fit(train)
    assert m.ensemble is not None
    X = test.X.copy()
    X[::7, 0] = np.nan
    np.testing.assert_allclose(
        m.ensemble.predict(X, backend=backend), m.ensemble.predict(X, "python")
    )
    assert m.ensemble.predict(np.zeros((0, train.n_features), dtype=np.float32)).shape == (0,)
    with pytest.raises(ValueError):
        m.ensemble.predict(np.zeros((2, 3)))


def test_ensemble_truncate_and_validation(medium: tuple[RankingData, ...]) -> None:
    train, _, test = medium
    m = LambdaMART(n_estimators=6, learning_rate=0.3).fit(train)
    ens = m.ensemble
    assert ens is not None
    t3 = ens.truncate(3)
    assert t3.n_trees == 3
    d = ens.to_dict()
    assert Ensemble.from_dict(d).n_trees == 6
    bad = dict(d, left=[5] * len(d["left"]))
    with pytest.raises(ValueError):
        Ensemble.from_dict(bad)
    bad = dict(d, split_feature=[99] * len(d["split_feature"]))
    with pytest.raises(ValueError):
        Ensemble.from_dict(bad)
    bad = dict(d, leaf_offsets=d["leaf_offsets"][:-1])
    with pytest.raises(ValueError):
        Ensemble.from_dict(bad)
    bad = dict(d, leaf_value=d["leaf_value"][:-1])
    with pytest.raises(ValueError):
        Ensemble.from_dict(bad)
    bad = dict(d, right=[-100] * len(d["right"]))
    with pytest.raises(ValueError):
        Ensemble.from_dict(bad)
    empty = Ensemble.empty(train.n_features)
    np.testing.assert_array_equal(empty.predict(test.X), np.zeros(test.n_docs))


def test_single_leaf_trees_predict_constant(small: RankingData) -> None:
    for backend in BACKENDS:
        m = LambdaMART(backend=backend, n_estimators=2, num_leaves=1).fit(small)
        assert m.ensemble is not None and m.ensemble.split_feature.size == 0
        p = m.predict(small)
        assert np.all(p == p[0])
        np.testing.assert_allclose(m.ensemble.predict(small.X, "python"), p)


def test_parameter_validation() -> None:
    for bad in (
        {"objective": "pairwise"},
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"feature_fraction": 1.5},
        {"max_bin": 1},
        {"max_depth": -1},
        {"lambda_l2": -1.0},
        {"sigma": 0.0},
        {"eval_at": 0},
    ):
        with pytest.raises(ValueError):
            GBDTParams(**bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LambdaMART(not_a_param=1)
    with pytest.raises(RuntimeError):
        LambdaMART().predict(np.zeros((1, 2)))


def test_feature_mismatch(medium: tuple[RankingData, ...]) -> None:
    train, _, _ = medium
    other = synthetic(5, n_features=3)
    with pytest.raises(ValueError):
        LambdaMART(n_estimators=1).fit(train, other)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError):
        LambdaMART(backend="gpu")
