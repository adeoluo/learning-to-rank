"""Unbiased learning to rank from click logs with inverse propensity weighting (IPW).

Under the position-based model a click on a document shown at rank ``r`` happens with
probability ``theta_r * P(R=1)``. Weighting each click by ``1 / theta_r`` makes the average
weighted click of a document an unbiased estimate of its attractiveness (Joachims et al., 2017;
Wang et al., 2016). This module turns a click log into per-document training targets:

* ``naive``: click-through rate, ``clicks / impressions`` (biased towards top positions);
* ``ipw``: ``sum(click / theta_rank) / impressions`` with given relative propensities.

Targets are linear in the clicks, so losses that are linear in the labels (the softmax listwise
loss, squared error) stay unbiased in expectation. Weights can be clipped (``max_weight``) to
trade a little bias for lower variance (Strehl et al., 2010).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.clicks import ClickLog
from learning_to_rank.data import RankingData


def click_targets(
    log: ClickLog,
    n_rows: int,
    propensity: NDArray[np.floating] | None = None,
    *,
    max_weight: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(target, impressions) per row; target is NaN for rows that were never shown."""

    rows, ranks, clicked = log.impressions()
    if propensity is None:
        weight = np.ones(log.top_k)
    else:
        prop = np.asarray(propensity, dtype=np.float64)
        if prop.shape != (log.top_k,) or not np.all(prop > 0):
            raise ValueError("propensity needs one positive value per rank")
        weight = 1.0 / prop
        if max_weight is not None:
            weight = np.minimum(weight, max_weight)
    shows = np.bincount(rows, minlength=n_rows).astype(np.float64)
    weighted = np.bincount(rows, weights=clicked * weight[ranks], minlength=n_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        target = np.where(shows > 0, weighted / shows, np.nan)
    return target, shows


def click_training_data(
    data: RankingData,
    log: ClickLog,
    propensity: NDArray[np.floating] | None = None,
    *,
    max_weight: float | None = None,
) -> RankingData:
    """The shown documents of ``data`` with click-derived labels (naive or IPW)."""

    target, shows = click_targets(log, data.n_docs, propensity, max_weight=max_weight)
    keep = shows > 0
    return RankingData(data.X[keep], target[keep].astype(np.float32), data.qid[keep])


def shown_with_true_labels(data: RankingData, log: ClickLog) -> RankingData:
    """The documents the log shows, with their true labels (the full-information skyline)."""

    _, shows = click_targets(log, data.n_docs)
    keep = shows > 0
    return RankingData(data.X[keep], data.y[keep], data.qid[keep])
