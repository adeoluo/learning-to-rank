"""Comparisons with reference libraries (uv sync --group validation)."""

from __future__ import annotations

import numpy as np
import pytest

from learning_to_rank import LambdaMART, RankingData, evaluate
from learning_to_rank.metrics import per_query

pytestmark = pytest.mark.validation


def test_ranx_map_mrr_ndcg(medium: tuple[RankingData, ...]) -> None:
    ranx = pytest.importorskip("ranx")
    _, _, test = medium
    s = np.random.default_rng(0).normal(size=test.n_docs)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    keep = []
    for q in range(test.n_queries):
        a, b = test.offsets[q], test.offsets[q + 1]
        if test.y[a:b].max() < 1:
            continue
        keep.append(q)
        qrels[f"q{q:05d}"] = {str(i): int(test.y[i]) for i in range(a, b)}
        run[f"q{q:05d}"] = {str(i): float(s[i]) for i in range(a, b)}
    ref = ranx.evaluate(
        ranx.Qrels(qrels), ranx.Run(run), ["map", "mrr", "ndcg_burges@10"], return_mean=False
    )
    ours = per_query(test.y, s, test.offsets, ks=(10,))
    np.testing.assert_allclose(ours["map"][keep], ref["map"], rtol=1e-9)
    np.testing.assert_allclose(ours["mrr"][keep], ref["mrr"], rtol=1e-9)
    np.testing.assert_allclose(ours["ndcg@10"][keep], ref["ndcg_burges@10"], rtol=1e-9)


def test_lightgbm_ndcg_metric_agrees(medium: tuple[RankingData, ...], openmp_safe: None) -> None:
    lgb = pytest.importorskip("lightgbm")
    train, valid, _ = medium
    evals: dict[str, dict[str, list[float]]] = {}
    ds = lgb.Dataset(train.X, train.y, group=train.group_sizes)
    dv = lgb.Dataset(valid.X, valid.y, group=valid.group_sizes, reference=ds)
    booster = lgb.train(
        {"objective": "lambdarank", "eval_at": [1, 5, 10], "verbose": -1, "num_threads": 2},
        ds,
        5,
        valid_sets=[dv],
        valid_names=["v"],
        callbacks=[lgb.record_evaluation(evals)],
    )
    m = evaluate(valid, booster.predict(valid.X))
    for k in (1, 5, 10):
        assert m[f"ndcg@{k}"] == pytest.approx(evals["v"][f"ndcg@{k}"][-1], abs=1e-6)


def test_quality_matches_lightgbm(medium: tuple[RankingData, ...], openmp_safe: None) -> None:
    lgb = pytest.importorskip("lightgbm")
    train, _, test = medium
    ours = LambdaMART(n_estimators=60).fit(train)
    ds = lgb.Dataset(train.X, train.y, group=train.group_sizes)
    ref = lgb.train({"objective": "lambdarank", "verbose": -1, "num_threads": 2}, ds, 60)
    a = evaluate(test, ours.predict(test))["ndcg@10"]
    b = evaluate(test, ref.predict(test.X))["ndcg@10"]
    assert abs(a - b) < 0.02, (a, b)


def test_xgboost_runs_on_same_data(medium: tuple[RankingData, ...], openmp_safe: None) -> None:
    xgb = pytest.importorskip("xgboost")
    train, _, test = medium
    d = xgb.DMatrix(train.X, train.y, qid=train.qid)
    b = xgb.train({"objective": "rank:ndcg", "tree_method": "hist", "nthread": 2}, d, 30)
    assert evaluate(test, b.predict(xgb.DMatrix(test.X)))["ndcg@10"] > 0.6
