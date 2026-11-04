# ADR-0002: A Rust histogram core with a numpy reference that produces the same trees

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

LambdaMART on MSLR-WEB10K builds 500 trees over 723,412 documents and 136 features per fold.
Each tree needs LambdaRank gradients for every training query (a pair loop over the top 30
positions of each list) and a histogram of gradient sums per feature bin for every leaf it
splits. In numpy, the per-query pair loop and the per-leaf histograms are Python loops over
thousands of queries and dozens of leaves; a single tree takes seconds, so a five-fold run would
take days. The point of the project is to compare a from-scratch implementation with LightGBM
and XGBoost on equal terms, so it has to be fast enough to run the same budget.

## Decision

Implement the hot paths in Rust (`rust/src/`), compiled into the package with maturin as
`learning_to_rank.learning_to_rank_rs` and used by default:

* `lambda.rs`: LambdaRank gradients and hessians per query in parallel (rayon), each query
  writing only its own slice; the pair loop reads ranked-order arrays and reuses per-thread
  buffers.
* `tree.rs`: leaf-wise histogram trees on a row-major `u8` bin matrix. Large leaves build the
  histogram from a fixed number of row chunks (a function of the leaf size only) whose partial
  histograms are added in chunk order; smaller leaves are built feature-parallel. The larger
  child of every split gets the parent minus the smaller child. Partitioning is stable.
* `binning.rs`, `boost.rs`, `metrics.rs`, `svmlight.rs`: bin edges, the boosting loop with
  validation NDCG, and a parallel LETOR parser.

Keep a readable numpy implementation of every step (`gbdt_reference.py`, `binning.py`,
`metrics.py`, `data.py`) that performs the same floating-point operations in the same order:
sequential sums via `cumsum`, the same chunking, the same tie rules.

## Alternatives considered

- **Numba or Cython.** Fewer moving parts in the build, but parallel loops with per-thread state
  and deterministic reductions are clumsier, and the portfolio already uses Rust cores.
- **Wrapping LightGBM's C API.** Fast, but then there is nothing of our own to compare.
- **Floating-point tolerance only.** Easier to write, but it hides real algorithmic drift (a
  different tie rule changes a tree silently).

## Consequences

- `tests/test_rust_parity.py` requires identical split features, thresholds and child pointers
  and leaf values within 1e-10 between Rust and numpy, with and without feature subsampling,
  both objectives, and the chunked histogram path; gradients agree within 1e-12.
- Training results are deterministic for any thread count.
- Building from source needs a Rust toolchain; CI installs one.
- The Rust trainer is 1.4 to 2.5 times slower than LightGBM on this machine (see the README);
  LightGBM approximates counts from hessians and uses a sigmoid lookup table, both of which we
  chose not to do to keep exact parity.
