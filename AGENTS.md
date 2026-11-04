# AGENTS.md

Working notes for coding agents and contributors in **Learning To Rank**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

LambdaMART with a Rust histogram core, neural rankers, click models and unbiased learning to rank, measured against LightGBM and XGBoost on MSLR-WEB10K and Istella-S.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/binning.rs`: Feature binning: per-feature bin upper edges fitted on a row sample, and the mapping from raw values to `u8` bin indices that the histogram tree learner works on.
- `rust/src/boost.rs`: Gradient boosting driver: LambdaMART (LambdaRank gradients) or squared-error regression, histogram trees, feature subsampling, validation NDCG and early stopping.
- `rust/src/lambda.rs`: LambdaRank gradients and hessians (Burges 2010), with the normalization choices LightGBM makes: pairs are formed between the top `truncation_level` positions of the current ranking and every position...
- `rust/src/lib.rs`: Rust core of `learning_to_rank`: LETOR parsing, feature binning, LambdaRank gradients, histogram tree learning, boosting and NDCG.
- `rust/src/metrics.rs`: Ranking metrics used inside training (validation NDCG for early stopping).
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `learning_to_rank_rs` extension module.
- `rust/src/svmlight.rs`: Parallel parser for the LETOR / SVMlight ranking format.
- `rust/src/tree.rs`: Histogram-based regression trees grown leaf-wise (best-first), as in LightGBM.
- `src/learning_to_rank/__init__.py`: Learning to rank: LambdaMART with a Rust histogram core, neural rankers, click models, unbiased learning to rank, ranking metrics, segment analysis and significance tests.
- `src/learning_to_rank/backend.py`: The Rust backend, compiled into the package on install and used by default.
- `src/learning_to_rank/binning.py`: Feature binning for the histogram tree learner.
- `src/learning_to_rank/cli.py`: Command line interface: ``python -m learning_to_rank <command>``.
- `src/learning_to_rank/clicks.py`: Click simulation and click models.
- `src/learning_to_rank/data.py`: Query-grouped ranking data and LETOR (SVMlight) input and output.
- `src/learning_to_rank/datasets.py`: Public learning-to-rank datasets: MSLR-WEB10K (five folds) and Istella-S.
- `src/learning_to_rank/gbdt.py`: LambdaMART: gradient boosted regression trees trained with LambdaRank gradients.
- `src/learning_to_rank/gbdt_reference.py`: Numpy reference implementation of the LambdaMART trainer.
- `src/learning_to_rank/metrics.py`: Vectorized ranking metrics: NDCG@k, MAP, MRR, ERR@k and precision@k, per query and averaged.
- `src/learning_to_rank/neural.py`: Neural rankers: an MLP scoring function trained with pointwise, pairwise or listwise losses.
- `src/learning_to_rank/observability.py`: Thread-safe service metrics in the Prometheus text exposition format.
- `src/learning_to_rank/segments.py`: Segment analysis: how ranking quality differs across groups of queries.
- `src/learning_to_rank/service.py`: HTTP ranking service (standard library).
- `src/learning_to_rank/significance.py`: Statistical significance for paired per-query comparisons of two rankers.
- `src/learning_to_rank/unbiased.py`: Unbiased learning to rank from click logs with inverse propensity weighting (IPW).

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (builds the Rust core) and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs numpy reference parity tests |
| `make validation` | Tests against LightGBM, XGBoost, scikit-learn and ranx |
| `make evaluation` | Every full-scale benchmark (hours on a laptop; needs make data), then the tables |
| `make report` | Print the README result tables from benchmarks/results |
| `make demos` | Re-record the demo GIFs with vhs (needs make data) |
| `make data` | Download and prepare MSLR-WEB10K (and Istella-S) under ../.bench-data/learning-to-rank |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
