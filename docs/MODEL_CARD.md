# Model / System Card - Learning To Rank

## Overview

- **What it is:** a library, CLI and HTTP service for learning to rank: LambdaMART on histogram
  trees (Rust core), MLP rankers with pointwise, pairwise and listwise losses (PyTorch), click
  models and inverse-propensity-weighted training from click logs, ranking metrics, significance
  tests and segment analysis.
- **Base models:** none; every model is trained from scratch on the data given.
- **Version and date:** 0.1.0, November 2026.

## Intended use

- Offline ranking experiments: training and comparing rankers on LETOR-format data, measuring
  NDCG, MAP, MRR and ERR with significance tests and per-segment breakdowns.
- Studying position bias and unbiased learning to rank with simulated clicks.
- Serving a trained model behind `/rank` for prototypes and demos.

## Out of scope

- Production search or ads ranking without further validation: models here were trained on
  public academic datasets with feature sets that do not exist elsewhere.
- Learning from real click logs without a measured propensity model: the IPW correction is only
  as good as the propensities, and real users show trust bias and context effects that the
  simulation does not model.

## Performance

Headline (details in [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md)): on MSLR-WEB10K, five folds,
our LambdaMART reaches test NDCG@10 0.5029 ± 0.0061 against 0.5042 ± 0.0056 for LightGBM's
`lambdarank` with the same settings, a difference that is not significant over the 10,000
queries (p = 0.10); XGBoost's `rank:ndcg` is slightly and significantly higher (0.5061,
p = 0.01). On Istella-S our LambdaMART and LightGBM are level (0.7603 and 0.7595, p = 0.20) and
XGBoost is lower (0.7554). The best MLP ranker (ListNet) reaches 0.4907 on MSLR. Tail queries
score lowest for every model (about 0.47 against 0.54 for torso queries); the per-segment
differences between our trainer and LightGBM are within 0.003 and not significant after Holm
adjustment. LightGBM trains 1.7 to 2.4 times faster.

## Risks and limitations

- Rankers amplify whatever the labels or clicks reward; click-trained models inherit position,
  presentation and popularity biases that the correction does not remove.
- Tree models extrapolate as constants outside the training range.
- Comparisons hold for the hyperparameters reported; neither library was tuned per dataset.

## Safety mitigations

- The HTTP service validates shapes and values, caps documents per request and body size, and
  exposes request metrics; it executes no code from inputs.
- Model files are JSON (trees) or weights-only PyTorch checkpoints, so loading a model cannot
  run code; the tree loader rejects malformed or cyclic trees.

## Environmental and cost footprint

All experiments ran on one Apple M1 laptop (8 cores, 8 GB) shared with other jobs: about 40
minutes for the five-fold tree comparison, about two hours for the 20 MLP runs on the shared
GPU, 19 minutes for the unbiased learning-to-rank study and about 20 minutes for Istella-S.
