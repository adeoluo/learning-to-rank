"""Training speed: thread scaling against LightGBM, and the Rust core against the numpy reference.

1. Thread scaling: 50 LambdaMART trees on MSLR-WEB10K fold 1 (723,412 documents) with 1, 2, 4
   and 8 threads, this package (RAYON_NUM_THREADS) and LightGBM (num_threads), same settings as
   gbdt_mslr.py. Each run is its own process.
2. Rust core against the numpy reference implementation on the first 300 training queries
   (5 trees), the per-phase time split of the Rust trainer on the full fold, and the LambdaRank
   gradient pass on all 723,412 documents.

    uv run --group validation python benchmarks/throughput.py

Results: benchmarks/results/throughput.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
from common import MSLR, ROOT, machine, write

TREES = 50


def one(lib: str, threads: int) -> dict[str, Any]:
    from learning_to_rank.datasets import load_mslr_fold

    train, _, _ = load_mslr_fold(MSLR, 1)
    t0 = time.perf_counter()
    if lib == "ours":
        from learning_to_rank import LambdaMART

        LambdaMART(n_estimators=TREES).fit(train)
    else:
        import lightgbm as lgb

        params = {"objective": "lambdarank", "num_threads": threads, "verbose": -1, "max_bin": 255}
        ds = lgb.Dataset(train.X, train.y, group=train.group_sizes, params=params)
        lgb.train(params, ds, TREES)
    return {"library": lib, "threads": threads, "seconds": time.perf_counter() - t0}


def reference_comparison() -> dict[str, Any]:
    from learning_to_rank import LambdaMART
    from learning_to_rank.backend import extension
    from learning_to_rank.binning import BinMapper
    from learning_to_rank.datasets import load_mslr_fold
    from learning_to_rank.gbdt_reference import lambdarank_gradients_reference

    train, _, _ = load_mslr_fold(MSLR, 1)
    small = train.subset(np.arange(300))
    out: dict[str, Any] = {"small_docs": small.n_docs}
    for backend in ("rust", "python"):
        t0 = time.perf_counter()
        LambdaMART(n_estimators=5, backend=backend).fit(small)
        out[f"train_5_trees_{backend}_seconds"] = time.perf_counter() - t0
    s = np.random.default_rng(0).normal(size=small.n_docs)
    t0 = time.perf_counter()
    lambdarank_gradients_reference(small.y.astype(np.float64), s, small.offsets)
    out["gradients_small_python_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    extension().lambdarank_gradients(small.y, s, small.offsets)
    out["gradients_small_rust_seconds"] = time.perf_counter() - t0
    s = np.random.default_rng(0).normal(size=train.n_docs)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        extension().lambdarank_gradients(train.y, s, train.offsets)
        times.append(time.perf_counter() - t0)
    out["gradients_full_rust_seconds_median"] = float(np.median(times))
    mapper = BinMapper.fit(train.X)
    bins = mapper.transform(train.X)
    booster = extension().Booster(bins, train.n_features, train.y, train.offsets, mapper.to_lists())
    booster.step(TREES)
    out["phase_seconds_per_tree"] = {k: v / TREES for k, v in booster.timings()}
    return out


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "one":
        print(json.dumps(one(sys.argv[2], int(sys.argv[3]))))
        return
    scaling = []
    for threads in (1, 2, 4, 8):
        for lib in ("ours", "lightgbm"):
            env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
            res = subprocess.run(
                [sys.executable, __file__, "one", lib, str(threads)],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
            )
            row = json.loads(res.stdout.strip().splitlines()[-1])
            scaling.append(row)
            print(row, flush=True)
    ref = reference_comparison()
    print(json.dumps(ref, indent=1))
    write(
        "throughput.json",
        {"trees": TREES, "scaling": scaling, "reference": ref, "machine": machine()},
    )


if __name__ == "__main__":
    main()
