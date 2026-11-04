# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-04

### Features

- LambdaMART on histogram trees with a Rust core, ranking metrics and MSLR folds (#1)
- Neural rankers with pointwise, pairwise and listwise losses (#2)
- Click simulation, click models and unbiased learning to rank (#3)
- Paired significance tests and segment analysis (#4)
- Command line interface and HTTP ranking service (#6)
- MSLR-WEB10K five-fold comparison with LightGBM and XGBoost (#8)
- Neural, unbiased learning to rank, segment, throughput and Istella-S benchmark results (#9)

### Performance

- Feature-parallel histograms for mid-size leaves and parallel stable partition (#5)
- LambdaRank pair loop over ranked-order arrays with per-thread scratch (#7)

### Documentation

- README with results, evaluation report and demos (#10)

### Maintenance

- Scaffold project
- Release tooling, agent and llms guides, changelog (#11)
