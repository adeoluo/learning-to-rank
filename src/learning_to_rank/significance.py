"""Statistical significance for paired per-query comparisons of two rankers.

* :func:`paired_t_test`: Student's t-test on per-query differences, with the t distribution
  computed here from the regularized incomplete beta function (continued fraction, Lentz's
  method), checked against SciPy in the tests.
* :func:`randomization_test`: Fisher's paired randomization (sign-flip) test, recommended for IR
  evaluation by Smucker, Allan and Carterette (CIKM 2007).
* :func:`bootstrap_ci`: percentile bootstrap interval of the mean difference.
* :func:`holm`: Holm-Bonferroni adjustment when several comparisons are reported together.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""

    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    """``P(|T| >= |t|)`` for Student's t with ``df`` degrees of freedom."""

    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(q: float, df: float) -> float:
    """Inverse CDF of Student's t (bisection on the two-sided tail), for ``0.5 < q < 1``."""

    if not 0.5 < q < 1.0:
        raise ValueError("q must be in (0.5, 1)")
    target = 2.0 * (1.0 - q)
    lo, hi = 0.0, 1.0
    while t_sf_two_sided(hi, df) > target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf_two_sided(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class TTest:
    t: float
    df: int
    p_value: float
    mean_diff: float
    ci_low: float
    ci_high: float


def _diff(a: NDArray[np.floating], b: NDArray[np.floating]) -> Array:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired samples must be 1-D arrays of equal length")
    if x.shape[0] < 2:
        raise ValueError("at least two paired observations are needed")
    out: Array = x - y
    return out


def paired_t_test(
    a: NDArray[np.floating], b: NDArray[np.floating], *, confidence: float = 0.95
) -> TTest:
    """Two-sided paired t-test of ``mean(a - b) = 0`` with a confidence interval of the mean."""

    d = _diff(a, b)
    n = d.shape[0]
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    df = n - 1
    if sd == 0.0:
        p = 1.0 if mean == 0.0 else 0.0
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        return TTest(t, df, p, mean, mean, mean)
    se = sd / math.sqrt(n)
    t = mean / se
    half = t_quantile(0.5 + confidence / 2.0, df) * se
    return TTest(t, df, t_sf_two_sided(t, df), mean, mean - half, mean + half)


def randomization_test(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
    chunk: int = 1000,
) -> float:
    """Two-sided p-value of Fisher's paired randomization test on the mean difference.

    Each permutation flips the sign of every per-query difference with probability 1/2; the
    p-value counts permutations at least as extreme as the observed mean, plus one for the
    observed assignment itself, so it is never 0.
    """

    d = _diff(a, b)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    while done < n_permutations:
        m = min(chunk, n_permutations - done)
        signs = rng.integers(0, 2, size=(m, d.shape[0]), dtype=np.int8) * 2 - 1
        stats = np.abs(signs @ d) / d.shape[0]
        extreme += int((stats >= observed - 1e-12).sum())
        done += m
    return (extreme + 1) / (n_permutations + 1)


def bootstrap_ci(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 1000,
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``mean(a - b)`` over queries."""

    d = _diff(a, b)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for s in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - s)
        idx = rng.integers(0, d.shape[0], size=(m, d.shape[0]))
        means[s : s + m] = d[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def holm(p_values: NDArray[np.floating] | list[float]) -> Array:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""

    p = np.asarray(p_values, dtype=np.float64)
    m = p.shape[0]
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adjusted
    return out


@dataclass(frozen=True)
class Comparison:
    """A against B on one metric over the same queries."""

    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    t_p_value: float
    randomization_p_value: float
    wins: int
    ties: int
    losses: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def compare(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    metric: str = "ndcg@10",
    n_permutations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Per-query paired comparison: means, t-test interval and p-value, randomization p-value,
    and the number of queries where A wins, ties or loses."""

    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t = paired_t_test(x, y)
    r = randomization_test(x, y, n_permutations=n_permutations, seed=seed)
    d = x - y
    return Comparison(
        metric=metric,
        n_queries=int(x.shape[0]),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        diff=t.mean_diff,
        ci_low=t.ci_low,
        ci_high=t.ci_high,
        t_p_value=t.p_value,
        randomization_p_value=r,
        wins=int((d > 1e-12).sum()),
        ties=int((np.abs(d) <= 1e-12).sum()),
        losses=int((d < -1e-12).sum()),
    )
