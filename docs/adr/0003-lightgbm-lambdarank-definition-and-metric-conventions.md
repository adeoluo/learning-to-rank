# ADR-0003: Follow LightGBM's LambdaRank definition and state every metric convention

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

"LambdaMART" names a family. Implementations differ in which pairs they use (all pairs, or
pairs touching the top k), whether they normalize lambdas per query, whether `|delta NDCG|` is
divided by the score gap, and how queries without relevant documents are scored. Published
NDCG numbers also differ on gain (`2^rel - 1` or `rel`), on ties and on empty queries. A
comparison with LightGBM is only meaningful if both sides optimize the same objective and are
scored by the same metric.

## Decision

* Gradients follow LightGBM's `lambdarank` objective: pairs between the top
  `truncation_level = 30` positions of the current ranking and every position below, `sigma =
  1`, `|delta NDCG|` against the ideal DCG at the truncation level, divided by `0.01 + |score
  gap|`, and the query's lambdas and hessians scaled by `log2(1 + S) / S`. We compute the sigmoid
  exactly rather than with LightGBM's lookup table.
* Defaults match LightGBM: 31 leaves grown leaf-wise, learning rate 0.1, `min_data_in_leaf` 20,
  `min_sum_hessian_in_leaf` 1e-3, no L2 penalty, 255 bins fitted on a 200,000-row sample.
* Metrics (`metrics.py`): gain `2^rel - 1`, discount `1 / log2(rank + 1)`, ties broken by stored
  document order, NDCG 1.0 for a query with no relevant document (LightGBM and XGBoost do the
  same), MAP and MRR with grade >= 1 as relevant, ERR with `max_grade = 4`. All three libraries'
  predictions are scored by this one module.

## Alternatives considered

- **Burges' original all-pairs LambdaRank.** Quadratic in list length (MSLR lists reach 1,251
  documents) and not what LightGBM runs.
- **Scoring each library with its own metric.** Small convention differences would then look
  like model differences.

## Consequences

- The validation tests check our NDCG@1/5/10 against LightGBM's own evaluation of the same
  predictions (agreement to 1e-6), MAP, MRR and NDCG against ranx, NDCG against scikit-learn, and
  that our trees match LightGBM's quality on synthetic data (NDCG@10 within 0.02).
- XGBoost's `rank:ndcg` uses its own pair sampling and an L2 penalty of 1 by default; we keep its
  defaults apart from tree size, learning rate, bins and threads and report it as is.
