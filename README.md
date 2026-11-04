<h1 align="center">Learning To Rank</h1>

<p align="center">
  <em>LambdaMART with a Rust histogram core, neural rankers, click models and unbiased learning to rank, measured against LightGBM and XGBoost on MSLR-WEB10K.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/learning-to-rank/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/learning-to-rank/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/learning-to-rank/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/learning-to-rank"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <img alt="Rust" src="https://img.shields.io/badge/rust-core-B7410E">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Ranking quality decides what users of a search engine or an ads system see first. This project
implements the standard learning-to-rank toolkit (LambdaMART on histogram trees, neural rankers,
click models and unbiased learning to rank from clicks) and measures it against LightGBM and
XGBoost on the public MSLR-WEB10K benchmark.

## Architecture

The package is a Python library with a Rust core compiled in: LETOR parsing, feature binning,
LambdaRank gradients and histogram tree learning run in Rust, and every one of them has a numpy
reference implementation that the parity tests compare against.

## Results

Benchmarks are added in a later change.

## Quickstart

```bash
make setup     # uv sync (builds the Rust core) and pre-commit
make test
```

```python
from learning_to_rank import LambdaMART, evaluate
from learning_to_rank.data import synthetic

train, test = synthetic(400, seed=1), synthetic(200, seed=2)
model = LambdaMART(n_estimators=100).fit(train)
print(evaluate(test, model.predict(test)))
```

## Usage

See the Quickstart; the command line and HTTP service are added in later changes.

## Design decisions

- Architecture decisions are recorded in [docs/adr/](docs/adr/).

## Testing

`make all` runs format, lint, type checks and the tests with an 80 percent coverage gate.
`make parity` compares the Rust core with the numpy reference; `make validation` compares
metrics and models with LightGBM, XGBoost, scikit-learn and ranx.

## Limitations

Work in progress.

## Roadmap

- Neural rankers, click models, unbiased learning to rank, segment analysis.

## License

Released under the [MIT License](LICENSE).
