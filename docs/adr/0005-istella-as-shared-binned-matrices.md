# ADR-0005: Train every library on the same 255-bin Istella-S matrices

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

Istella-S has 3,408,630 documents with 220 features: 3.0 GB as float32, 5.3 GB as LETOR text.
The machine has 8 GB of RAM shared with other work and a 3 GB per-job budget, so neither our
trainer nor LightGBM nor XGBoost can load the raw float matrix of the training split next to
their own binned copies.

## Decision

Convert the archive once, streaming it twice from the `.tar.gz` (`prepare_istella`): fit 255-bin
edges on every tenth training row, then write each split as a row-major `uint8` matrix of bin
indices plus labels and query ids (750 MB in total). All three libraries train on these bin
indices: ours through `LambdaMART.fit_binned` (the bin index is the feature value), LightGBM
through `lgb.Sequence` and XGBoost through a `DataIter`, both in batches of 100,000 rows.

## Alternatives considered

- **float16 features.** Halves memory but rounds large count features and is not what any
  library bins on.
- **Subsampling queries.** Would fit, but the point of the second dataset is scale.

## Consequences

- The three libraries see identical information; any difference is in the learners, not the
  input. LightGBM and XGBoost re-bin the bin indices, which is lossless with at most 255 distinct
  values per feature.
- Results are not directly comparable with papers that train on raw Istella-S features, though
  LightGBM itself bins to 255 values by default.
