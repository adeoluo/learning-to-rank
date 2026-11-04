# ADR-0006: Semi-synthetic clicks with swap interventions, and IPW targets linear in clicks

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Unbiased learning to rank needs click logs with known position bias to measure whether a
correction works. No public log ships the ground truth, so the standard protocol (Joachims et
al., 2017; Ai et al., 2018) simulates users on a labeled dataset. Two practical problems follow:
the position-based model cannot separate position effects from relevance when every document is
always shown at the same rank, and many unbiased losses are only unbiased when they are linear in
the click signal.

## Decision

* Simulate clicks (`clicks.py`) under the position-based model or the cascade model on the top 10
  of a weak production ranker (LambdaMART on 60 training queries), with attractiveness
  `0.1 + 0.9 (2^y - 1) / 15`. A configurable fraction of sessions (10 percent in the benchmarks)
  swaps rank 1 with a random rank, which gives the swap estimator of Joachims et al. and lets EM
  identify the examination curve.
* Fit click models from the log: PBM by EM on per (document, rank) sufficient statistics, the
  cascade model by maximum likelihood, and CTR by rank as a baseline; report held-out
  log-likelihood and perplexity per rank.
* Train on per-document targets `sum(click / theta_rank) / impressions` (`unbiased.py`), which
  are unbiased estimates of attractiveness under the PBM, with the squared-error objective of our
  LambdaMART (linear in the target, so the correction stays unbiased). Compare naive CTR,
  IPW with true, swap-estimated and EM-estimated propensities, and the same learner on the true
  attractiveness of the same shown documents.

## Alternatives considered

- **IPW-weighted LambdaRank.** The lambda weights depend on `2^target`, so the estimate is not
  unbiased; kept out of the headline.
- **Dual learning algorithm or regression EM** (Ai et al., 2018; Wang et al., 2018). Stronger
  estimators, more machinery; left on the roadmap.

## Consequences

- The skyline and the IPW models share the learner and the documents, so the gap between them
  measures only the click noise and the propensity estimate.
- Results hold for simulated users; real logs add trust bias, position-dependent noise and
  context the simulation does not have.
