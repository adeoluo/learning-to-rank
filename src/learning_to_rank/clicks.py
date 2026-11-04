"""Click simulation and click models.

Simulation follows the standard semi-synthetic protocol for unbiased learning to rank (Joachims
et al., 2017; Ai et al., 2018): a production ranker orders each query's documents, users see the
top ``k``, and clicks are drawn from a click model whose attractiveness comes from the true
relevance grade, ``P(R=1 | y) = eps + (1 - eps) (2^y - 1) / (2^max_grade - 1)``.

* Position-based model (PBM): a document at rank ``r`` is examined with probability
  ``theta_r = (1 / r)^eta`` independently of the others, and clicked if examined and attractive.
* Cascade model: the user scans from the top, clicks the first attractive document and stops.

Optionally a fraction of sessions applies a random swap intervention (the document at rank 1
trades places with the one at a uniformly chosen rank ``k``), which is what the swap estimator of
position bias needs and also lets EM separate position effects from relevance.

Fitted click models (PBM by expectation maximization, cascade by maximum likelihood, and a
click-through-rate-per-rank baseline) report log-likelihood and perplexity per rank on held-out
sessions, as in Chuklin et al., *Click Models for Web Search* (2015).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.data import RankingData

ClickModelName = Literal["pbm", "cascade"]
_EPS = 1e-6


@dataclass(frozen=True)
class ClickConfig:
    model: ClickModelName = "pbm"
    eta: float = 1.0
    epsilon: float = 0.1
    max_grade: float = 4.0
    top_k: int = 10
    swap_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in ("pbm", "cascade"):
            raise ValueError("model must be 'pbm' or 'cascade'")
        if self.eta < 0 or not 0 <= self.epsilon <= 1 or self.top_k < 1:
            raise ValueError("invalid eta, epsilon or top_k")
        if not 0 <= self.swap_fraction <= 1:
            raise ValueError("swap_fraction must be in [0, 1]")


def attractiveness(
    y: NDArray[np.floating], epsilon: float, max_grade: float
) -> NDArray[np.float64]:
    """Click probability of an examined document with grade ``y``."""

    scale = np.exp2(max_grade) - 1.0
    out: NDArray[np.float64] = (
        epsilon + (1.0 - epsilon) * (np.exp2(np.asarray(y, np.float64)) - 1.0) / scale
    )
    return out


def pbm_propensity(top_k: int, eta: float) -> NDArray[np.float64]:
    """Examination probability per rank 1..top_k under the PBM."""

    out: NDArray[np.float64] = (1.0 / np.arange(1, top_k + 1, dtype=np.float64)) ** eta
    return out


def top_k_rows(data: RankingData, scores: NDArray[np.floating], k: int) -> NDArray[np.int64]:
    """Row indices of each query's top ``k`` by descending score (ties keep order), -1 padded."""

    out = np.full((data.n_queries, k), -1, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    qi = data.query_index()
    by_score = np.argsort(-s, kind="stable")
    order = by_score[np.argsort(qi[by_score], kind="stable")]
    pos = np.arange(data.n_docs) - data.offsets[qi]
    keep = pos < k
    out[qi[keep], pos[keep]] = order[keep]
    return out


@dataclass
class ClickLog:
    """Sessions: the query position, the rows shown at ranks 1..k (-1 if fewer), the clicks, and
    which sessions carried a swap intervention (the rank swapped with rank 1, else 0)."""

    query: NDArray[np.int64]
    docs: NDArray[np.int64]
    clicks: NDArray[np.bool_]
    swap_rank: NDArray[np.int64]

    @property
    def n_sessions(self) -> int:
        return int(self.query.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.docs.shape[1])

    def split(self, fraction: float, seed: int = 0) -> tuple[ClickLog, ClickLog]:
        """Random split of sessions into (first, second) with ``fraction`` in the first."""

        perm = np.random.default_rng(seed).permutation(self.n_sessions)
        cut = round(fraction * self.n_sessions)
        return self.take(perm[:cut]), self.take(perm[cut:])

    def take(self, idx: NDArray[np.int64]) -> ClickLog:
        return ClickLog(self.query[idx], self.docs[idx], self.clicks[idx], self.swap_rank[idx])

    def impressions(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        """Flattened (row, rank0, clicked) over all shown positions."""

        shown = self.docs >= 0
        ranks = np.broadcast_to(np.arange(self.top_k), self.docs.shape)
        return self.docs[shown], ranks[shown].astype(np.int64), self.clicks[shown]


def simulate_clicks(
    data: RankingData,
    scores: NDArray[np.floating],
    n_sessions: int,
    config: ClickConfig | None = None,
    *,
    seed: int = 0,
    query_weights: NDArray[np.floating] | None = None,
    chunk: int = 250_000,
) -> ClickLog:
    """Simulate ``n_sessions`` sessions on ``data`` ranked by ``scores`` (the production ranker)."""

    cfg = config or ClickConfig()
    rng = np.random.default_rng(seed)
    k = cfg.top_k
    ranking = top_k_rows(data, scores, k)
    attract = attractiveness(data.y, cfg.epsilon, cfg.max_grade)
    theta = pbm_propensity(k, cfg.eta)
    p = None
    if query_weights is not None:
        w = np.asarray(query_weights, dtype=np.float64)
        if w.shape != (data.n_queries,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("query_weights must be non-negative with one weight per query")
        p = w / w.sum()
    queries, docs, clicks, swaps = [], [], [], []
    for start in range(0, n_sessions, chunk):
        m = min(chunk, n_sessions - start)
        q = rng.choice(data.n_queries, size=m, p=p).astype(np.int64)
        shown = ranking[q].copy()
        swap = np.zeros(m, dtype=np.int64)
        if cfg.swap_fraction > 0 and k > 1:
            do = rng.random(m) < cfg.swap_fraction
            r = rng.integers(1, k, size=m)
            valid = do & (shown[np.arange(m), r] >= 0)
            idx = np.flatnonzero(valid)
            first = shown[idx, 0].copy()
            shown[idx, 0] = shown[idx, r[idx]]
            shown[idx, r[idx]] = first
            swap[idx] = r[idx]
        a = np.where(shown >= 0, attract[np.maximum(shown, 0)], 0.0)
        u = rng.random((m, k))
        if cfg.model == "pbm":
            c = u < theta * a
        else:
            attractive = u < a
            any_click = attractive.any(axis=1)
            first_click = np.argmax(attractive, axis=1)
            c = np.zeros((m, k), dtype=bool)
            c[np.flatnonzero(any_click), first_click[any_click]] = True
        queries.append(q)
        docs.append(shown)
        clicks.append(c)
        swaps.append(swap)
    if not queries:
        empty = np.zeros((0, k), dtype=np.int64)
        return ClickLog(np.zeros(0, np.int64), empty, empty.astype(bool), np.zeros(0, np.int64))
    return ClickLog(
        np.concatenate(queries), np.concatenate(docs), np.concatenate(clicks), np.concatenate(swaps)
    )


def swap_propensity(log: ClickLog) -> NDArray[np.float64]:
    """Relative examination per rank from swap interventions (Joachims et al., 2017).

    In sessions that swapped rank 1 with rank ``k``, the production top document sits at ``k``;
    in sessions without a swap it sits at rank 1. Its click rate at ``k`` divided by its click
    rate at rank 1 estimates ``theta_k / theta_1`` because its relevance cancels.
    """

    k = log.top_k
    base = log.swap_rank == 0
    if not base.any():
        raise ValueError("no sessions without an intervention")
    ctr_top = log.clicks[base, 0].mean()
    if ctr_top <= 0:
        raise ValueError("no clicks at rank 1")
    out = np.full(k, np.nan)
    out[0] = 1.0
    for r in range(1, k):
        sel = log.swap_rank == r
        if sel.any():
            out[r] = log.clicks[sel, r].mean() / ctr_top
    return out


@dataclass
class FittedClickModel:
    """Parameters of a fitted click model: per-rank examination ``theta`` and per-row
    attractiveness ``alpha`` (NaN for rows never shown, which then use ``alpha_prior``)."""

    name: str
    theta: NDArray[np.float64]
    alpha: NDArray[np.float64]
    alpha_prior: float
    iterations: int = 0
    history: list[float] = field(default_factory=list)

    def relative_theta(self) -> NDArray[np.float64]:
        """Examination relative to rank 1 (what inverse propensity weighting needs)."""

        out: NDArray[np.float64] = self.theta / self.theta[0]
        return out

    def _alpha(self, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        a = self.alpha[np.maximum(rows, 0)]
        return np.where(np.isnan(a), self.alpha_prior, a)

    def click_probabilities(self, log: ClickLog) -> NDArray[np.float64]:
        """``P(C_r = 1 | clicks above r)`` for every session and rank (the model's conditional)."""

        a = self._alpha(log.docs)
        if self.name == "cascade":
            clicked_before = np.cumsum(log.clicks, axis=1) - log.clicks > 0
            p = np.where(clicked_before, 0.0, a)
        elif self.name == "pbm":
            p = self.theta[None, :] * a
        else:  # ctr per rank
            p = np.broadcast_to(self.theta, log.docs.shape).copy()
        return np.where(log.docs >= 0, p, 0.0)

    def evaluate(self, log: ClickLog) -> dict[str, float | list[float]]:
        """Mean log-likelihood per shown position and perplexity per rank (lower is better)."""

        p = np.clip(self.click_probabilities(log), _EPS, 1 - _EPS)
        ll = np.where(log.clicks, np.log2(p), np.log2(1 - p))
        shown = log.docs >= 0
        per_rank = [
            float(2.0 ** (-ll[shown[:, r], r].mean())) if shown[:, r].any() else float("nan")
            for r in range(log.top_k)
        ]
        return {
            "log_likelihood": float((ll[shown] * np.log(2.0)).mean()),
            "perplexity": float(np.nanmean(per_rank)),
            "perplexity_per_rank": per_rank,
        }


def fit_ctr(log: ClickLog, n_rows: int) -> FittedClickModel:
    """Baseline: click probability depends on the rank only."""

    _, ranks, clicked = log.impressions()
    shows = np.bincount(ranks, minlength=log.top_k)
    clicks = np.bincount(ranks, weights=clicked, minlength=log.top_k)
    theta = clicks / np.maximum(shows, 1)
    return FittedClickModel(
        "ctr", theta, np.full(n_rows, np.nan), float(clicked.mean()) if clicked.size else 0.0
    )


def fit_cascade(log: ClickLog, n_rows: int) -> FittedClickModel:
    """Cascade model: alpha = clicks / examinations, where a session examines every rank up to
    and including its first click (all ranks when nothing was clicked)."""

    any_click = log.clicks.any(axis=1)
    first = np.where(any_click, np.argmax(log.clicks, axis=1), log.top_k - 1)
    examined = (np.arange(log.top_k)[None, :] <= first[:, None]) & (log.docs >= 0)
    first_click = log.clicks & (np.cumsum(log.clicks, axis=1) == 1)
    rows = log.docs[examined]
    exams = np.bincount(rows, minlength=n_rows).astype(np.float64)
    clicks = np.bincount(rows, weights=first_click[examined], minlength=n_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(exams > 0, clicks / exams, np.nan)
    prior = float(clicks.sum() / max(exams.sum(), 1.0))
    return FittedClickModel("cascade", np.ones(log.top_k), alpha, prior)


def fit_pbm(
    log: ClickLog,
    n_rows: int,
    *,
    iterations: int = 2000,
    tol: float = 1e-7,
    theta_init: float = 0.5,
    alpha_init: float = 0.2,
) -> FittedClickModel:
    """Position-based model by expectation maximization (Chuklin et al., 2015, section 4.2).

    E-step for an unclicked impression with current ``theta`` and ``alpha``:
    ``P(E=1 | C=0) = theta (1 - alpha) / (1 - theta alpha)`` and
    ``P(A=1 | C=0) = alpha (1 - theta) / (1 - theta alpha)``; clicked impressions have both
    equal to 1. The M-step averages these posteriors per rank and per document. ``theta`` and
    ``alpha`` are identified only up to a common scale (and only when the same documents appear
    at several ranks, for example through swap interventions), so IPS uses
    :meth:`FittedClickModel.relative_theta`.
    """

    rows_all, ranks_all, clicked_all = log.impressions()
    # Sufficient statistics: shows and clicks per distinct (row, rank) pair.
    key = rows_all * log.top_k + ranks_all
    uniq, inverse = np.unique(key, return_inverse=True)
    shows = np.bincount(inverse).astype(np.float64)
    clicks = np.bincount(inverse, weights=clicked_all.astype(np.float64))
    shown_rows, rows = np.unique(uniq // log.top_k, return_inverse=True)
    ranks = uniq % log.top_k
    n_shown = shown_rows.shape[0]
    shows_row = np.bincount(rows, weights=shows, minlength=n_shown)
    shows_rank = np.bincount(ranks, weights=shows, minlength=log.top_k)
    theta = np.full(log.top_k, theta_init)
    alpha = np.full(n_shown, alpha_init)
    history: list[float] = []
    for _ in range(iterations):
        t, a = theta[ranks], alpha[rows]
        denom = np.clip(1.0 - t * a, _EPS, None)
        no_click = shows - clicks
        post_e = clicks + no_click * t * (1 - a) / denom
        post_a = clicks + no_click * a * (1 - t) / denom
        new_theta = np.bincount(ranks, weights=post_e, minlength=log.top_k) / np.maximum(
            shows_rank, 1
        )
        new_alpha = np.bincount(rows, weights=post_a, minlength=n_shown) / np.maximum(shows_row, 1)
        p = np.clip(new_theta[ranks] * new_alpha[rows], _EPS, 1 - _EPS)
        ll = (clicks * np.log(p) + no_click * np.log(1 - p)).sum() / max(shows.sum(), 1.0)
        history.append(float(ll))
        delta = max(np.abs(new_theta - theta).max(), np.abs(new_alpha - alpha).max())
        theta, alpha = new_theta, new_alpha
        if delta < tol:
            break
    full = np.full(n_rows, np.nan)
    full[shown_rows] = alpha
    prior = float(alpha.mean()) if n_shown else alpha_init
    return FittedClickModel("pbm", theta, full, prior, iterations=len(history), history=history)
