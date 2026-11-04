from __future__ import annotations

import numpy as np
import pytest

from learning_to_rank import LambdaMART, RankingData, evaluate
from learning_to_rank.clicks import (
    ClickConfig,
    attractiveness,
    fit_cascade,
    fit_ctr,
    fit_pbm,
    pbm_propensity,
    simulate_clicks,
    swap_propensity,
    top_k_rows,
)
from learning_to_rank.data import synthetic
from learning_to_rank.unbiased import click_targets, click_training_data, shown_with_true_labels


@pytest.fixture(scope="module")
def world() -> tuple[RankingData, np.ndarray]:
    data = synthetic(300, (10, 30), 8, seed=7)
    prod = np.random.default_rng(0).normal(size=data.n_docs) + 0.5 * data.y
    return data, prod


def test_attractiveness_and_propensity() -> None:
    a = attractiveness(np.array([0, 4, 2]), 0.1, 4)
    np.testing.assert_allclose(a, [0.1, 1.0, 0.1 + 0.9 * 3 / 15])
    np.testing.assert_allclose(pbm_propensity(3, 1.0), [1, 0.5, 1 / 3])
    np.testing.assert_allclose(pbm_propensity(3, 0.0), [1, 1, 1])


def test_config_validation() -> None:
    for bad in (
        {"model": "dbn"},
        {"eta": -1},
        {"epsilon": 2},
        {"top_k": 0},
        {"swap_fraction": 1.5},
    ):
        with pytest.raises(ValueError):
            ClickConfig(**bad)  # type: ignore[arg-type]


def test_top_k_rows_pads_short_lists() -> None:
    d = RankingData.from_arrays(np.zeros((5, 1)), [0, 1, 2, 0, 1], [1, 1, 1, 2, 2])
    r = top_k_rows(d, np.array([0.1, 0.9, 0.5, 0.0, 0.0]), 3)
    assert r.tolist() == [[1, 2, 0], [3, 4, -1]]


def test_pbm_click_rates_follow_propensity(world: tuple[RankingData, np.ndarray]) -> None:
    data, prod = world
    log = simulate_clicks(data, prod, 60_000, ClickConfig(epsilon=1.0), seed=1)
    # With epsilon = 1 every examined document is clicked, so CTR per rank equals theta.
    np.testing.assert_allclose(log.clicks.mean(axis=0), pbm_propensity(10, 1.0), atol=0.01)
    ctr = fit_ctr(log, data.n_docs)
    np.testing.assert_allclose(ctr.theta, pbm_propensity(10, 1.0), atol=0.01)


def test_cascade_clicks_at_most_once(world: tuple[RankingData, np.ndarray]) -> None:
    data, prod = world
    log = simulate_clicks(data, prod, 5000, ClickConfig(model="cascade"), seed=2)
    assert log.clicks.sum(axis=1).max() <= 1
    cas = fit_cascade(log, data.n_docs)
    shown = np.isfinite(cas.alpha)
    true = attractiveness(data.y, 0.1, 4)
    corr = np.corrcoef(cas.alpha[shown], true[shown])[0, 1]
    assert corr > 0.5
    ev = cas.evaluate(log)
    assert ev["log_likelihood"] < 0 and ev["perplexity"] >= 1  # type: ignore[operator]


def test_swap_and_em_recover_position_bias(world: tuple[RankingData, np.ndarray]) -> None:
    data, prod = world
    cfg = ClickConfig(eta=1.0, swap_fraction=0.3)
    log = simulate_clicks(data, prod, 150_000, cfg, seed=3)
    assert (log.swap_rank > 0).any()
    true = pbm_propensity(10, 1.0)
    np.testing.assert_allclose(swap_propensity(log), true, atol=0.06)
    pbm = fit_pbm(log, data.n_docs, iterations=500)
    np.testing.assert_allclose(pbm.relative_theta(), true, atol=0.06)
    assert pbm.history[-1] >= pbm.history[0]
    fit, held = log.split(0.7, seed=0)
    m_pbm = fit_pbm(fit, data.n_docs, iterations=200)
    m_ctr = fit_ctr(fit, data.n_docs)
    m_cas = fit_cascade(fit, data.n_docs)
    ll = {m.name: m.evaluate(held)["log_likelihood"] for m in (m_pbm, m_ctr, m_cas)}
    assert ll["pbm"] > ll["ctr"] > ll["cascade"]  # type: ignore[operator]


def test_query_weights_and_errors(world: tuple[RankingData, np.ndarray]) -> None:
    data, prod = world
    w = np.zeros(data.n_queries)
    w[5] = 1.0
    log = simulate_clicks(data, prod, 100, seed=4, query_weights=w)
    assert set(log.query.tolist()) == {5}
    with pytest.raises(ValueError):
        simulate_clicks(data, prod, 10, query_weights=np.zeros(data.n_queries))
    empty = simulate_clicks(data, prod, 0)
    assert empty.n_sessions == 0
    no_base = simulate_clicks(data, prod, 50, ClickConfig(swap_fraction=1.0), seed=1)
    with pytest.raises(ValueError):
        swap_propensity(no_base)
    nothing = simulate_clicks(data, prod, 20, ClickConfig(epsilon=0.0, eta=50), seed=1)
    nothing.clicks[:] = False
    with pytest.raises(ValueError):
        swap_propensity(nothing)


def test_ipw_targets() -> None:
    from learning_to_rank.clicks import ClickLog

    log = ClickLog(
        query=np.array([0, 0]),
        docs=np.array([[0, 1], [1, 0]]),
        clicks=np.array([[True, False], [False, True]]),
        swap_rank=np.array([0, 1]),
    )
    naive, shows = click_targets(log, 3)
    assert shows.tolist() == [2, 2, 0]
    assert naive[:2].tolist() == [1.0, 0.0] and np.isnan(naive[2])
    ipw, _ = click_targets(log, 3, np.array([1.0, 0.5]))
    assert ipw[:2].tolist() == [1.5, 0.0]
    clipped, _ = click_targets(log, 3, np.array([1.0, 0.5]), max_weight=1.5)
    assert clipped[0] == 1.25
    with pytest.raises(ValueError):
        click_targets(log, 3, np.array([1.0, 0.0]))


def test_ipw_beats_naive_on_synthetic() -> None:
    train, test = synthetic(500, (20, 40), 10, seed=1), synthetic(200, (20, 40), 10, seed=2)
    rng = np.random.default_rng(0)
    prod = LambdaMART(n_estimators=10).fit(
        train.subset(rng.choice(train.n_queries, 20, replace=False))
    )
    log = simulate_clicks(train, prod.predict(train), 100_000, ClickConfig(eta=2.0), seed=5)
    scores = {}
    for name, prop in (("naive", None), ("ipw", pbm_propensity(10, 2.0))):
        d = click_training_data(train, log, prop)
        m = LambdaMART(objective="regression", n_estimators=60, learning_rate=0.1).fit(d)
        scores[name] = evaluate(test, m.predict(test))["ndcg@10"]
    sky = shown_with_true_labels(train, log)
    assert sky.n_docs == click_training_data(train, log).n_docs
    assert scores["ipw"] > scores["naive"]
