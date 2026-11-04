from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_to_rank.significance import (
    betainc,
    bootstrap_ci,
    compare,
    holm,
    paired_t_test,
    randomization_test,
    t_quantile,
    t_sf_two_sided,
)


def test_known_t_values() -> None:
    # Two-sided p of t = 2.228 with 10 df is 0.05 (textbook table).
    assert t_sf_two_sided(2.228, 10) == pytest.approx(0.05, abs=2e-4)
    assert t_quantile(0.975, 10) == pytest.approx(2.228, abs=1e-3)
    assert t_sf_two_sided(0.0, 5) == pytest.approx(1.0)
    assert t_sf_two_sided(float("inf"), 5) == 0.0
    assert np.isnan(t_sf_two_sided(float("nan"), 5))
    assert betainc(2, 3, 0.0) == 0.0 and betainc(2, 3, 1.0) == 1.0
    with pytest.raises(ValueError):
        betainc(1, 1, 1.5)
    with pytest.raises(ValueError):
        t_quantile(0.2, 3)


@settings(max_examples=60, deadline=None)
@given(t=st.floats(-50, 50), df=st.integers(1, 5000))
def test_t_distribution_matches_scipy(t: float, df: int) -> None:
    stats = pytest.importorskip("scipy.stats")
    assert t_sf_two_sided(t, df) == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-7, abs=1e-14)


def test_paired_t_test_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, size=300)
    b = a - rng.normal(0.01, 0.05, size=300)
    ours = paired_t_test(a, b)
    ref = stats.ttest_rel(a, b)
    assert ours.t == pytest.approx(ref.statistic, rel=1e-10)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    ci = ref.confidence_interval(0.95)
    assert (ours.ci_low, ours.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-6)


def test_degenerate_differences() -> None:
    a = np.ones(5)
    assert paired_t_test(a, a).p_value == 1.0
    assert paired_t_test(a + 1, a).p_value == 0.0
    with pytest.raises(ValueError):
        paired_t_test(np.ones(1), np.ones(1))
    with pytest.raises(ValueError):
        paired_t_test(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        randomization_test(a, a, n_permutations=0)


def test_randomization_test_behaviour() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    same = randomization_test(a, a + rng.normal(scale=1e-9, size=200), n_permutations=2000)
    assert same > 0.05
    shifted = randomization_test(a + 0.5, a + rng.normal(scale=0.1, size=200), n_permutations=2000)
    assert shifted == pytest.approx(1 / 2001)
    exact = randomization_test(
        np.array([1.0, 2.0]), np.array([0.0, 0.0]), n_permutations=4000, seed=2
    )
    assert exact == pytest.approx(0.5, abs=0.05)  # 2 of the 4 sign patterns are as extreme


def test_randomization_agrees_with_t_test_on_large_samples() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, size=2000)
    b = a + rng.normal(-0.04, 1.0, size=2000)
    p_t = paired_t_test(a, b).p_value
    p_r = randomization_test(a, b, n_permutations=20_000, seed=4)
    assert p_r == pytest.approx(p_t, abs=0.02)


def test_bootstrap_ci_contains_mean() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.normal(1, 1, 500), rng.normal(0.8, 1, 500)
    lo, hi = bootstrap_ci(a, b, n_resamples=2000)
    assert lo < (a - b).mean() < hi


def test_holm() -> None:
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5]), [0.5])
    assert (holm([0.9, 0.8]) <= 1).all()


def test_compare_counts() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.2])
    b = np.array([0.4, 0.6, 0.9, 0.1])
    c = compare(a, b, n_permutations=500)
    assert (c.wins, c.ties, c.losses) == (2, 1, 1)
    assert c.diff == pytest.approx(0.0)
    assert c.to_dict()["metric"] == "ndcg@10"
