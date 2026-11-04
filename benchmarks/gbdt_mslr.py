"""LambdaMART on MSLR-WEB10K, five folds: this package against LightGBM and XGBoost.

Every library trains on the same fold files with the same budget: 500 trees, learning rate 0.1,
31 leaves grown leaf-wise, at most 255 bins, all 8 threads, no early stopping. Each run is a
separate process (so peak memory is per library and the OpenMP runtimes never mix), and test
metrics are computed by `learning_to_rank.metrics` for all three.

    uv run --group validation python benchmarks/gbdt_mslr.py all          # 15 runs
    uv run --group validation python benchmarks/gbdt_mslr.py run ours 1   # one run

Results: benchmarks/results/gbdt_mslr_<lib>_fold<k>.json
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
from common import MSLR, ROOT, Timer, machine, peak_rss_mb, write

TREES, LR, LEAVES, MIN_DATA, MAX_BIN, THREADS = 500, 0.1, 31, 20, 255, 8
LIBS = ("ours", "lightgbm", "xgboost")


def run(lib: str, fold: int) -> dict[str, Any]:
    from learning_to_rank.datasets import load_mslr_fold
    from learning_to_rank.metrics import evaluate, per_query

    train, _, test = load_mslr_fold(MSLR, fold)
    construct = 0.0
    if lib == "ours":
        from learning_to_rank import LambdaMART

        model = LambdaMART(
            n_estimators=TREES,
            learning_rate=LR,
            num_leaves=LEAVES,
            min_data_in_leaf=MIN_DATA,
            max_bin=MAX_BIN,
        )
        with Timer() as t:
            model.fit(train)
        construct = model.result.binning_seconds
        with Timer() as tp:
            scores = model.predict(test)
        version = "this package"
    elif lib == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "lambdarank",
            "learning_rate": LR,
            "num_leaves": LEAVES,
            "min_data_in_leaf": MIN_DATA,
            "max_bin": MAX_BIN,
            "num_threads": THREADS,
            "verbose": -1,
            "seed": 0,
        }
        with Timer() as t:
            with Timer() as tc:
                ds = lgb.Dataset(
                    train.X, train.y, group=train.group_sizes, params=params
                ).construct()
            booster = lgb.train(params, ds, TREES)
        construct = tc.seconds
        with Timer() as tp:
            scores = booster.predict(test.X, num_threads=THREADS)
        version = lgb.__version__
    elif lib == "xgboost":
        import xgboost as xgb

        params = {
            "objective": "rank:ndcg",
            "eta": LR,
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_leaves": LEAVES,
            "max_depth": 0,
            "max_bin": MAX_BIN,
            "nthread": THREADS,
            "seed": 0,
        }
        with Timer() as t:
            with Timer() as tc:
                dtrain = xgb.QuantileDMatrix(
                    train.X, train.y, qid=train.query_index(), max_bin=MAX_BIN
                )
            booster = xgb.train(params, dtrain, TREES)
        construct = tc.seconds
        with Timer() as tp:
            scores = booster.predict(xgb.DMatrix(test.X, nthread=THREADS))
        version = xgb.__version__
    else:
        raise SystemExit(f"unknown library {lib}")
    scores = np.asarray(scores, dtype=np.float64)
    pq = per_query(test.y, scores, test.offsets)
    return {
        "library": lib,
        "version": version,
        "fold": fold,
        "train_docs": train.n_docs,
        "train_queries": train.n_queries,
        "test_docs": test.n_docs,
        "test_queries": test.n_queries,
        "trees": TREES,
        "fit_seconds": t.seconds,
        "construct_seconds": construct,
        "predict_seconds": tp.seconds,
        "peak_rss_mb": peak_rss_mb(),
        "metrics": evaluate(test, scores),
        "per_query": {"qid": test.query_ids.tolist(), "ndcg@10": pq["ndcg@10"].round(6).tolist()},
        "machine": machine(),
    }


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        lib, fold = sys.argv[2], int(sys.argv[3])
        res = run(lib, fold)
        path = write(f"gbdt_mslr_{lib}_fold{fold}.json", res)
        m = res["metrics"]
        print(
            f"{lib:<9} fold {fold}  ndcg@10 {m['ndcg@10']:.4f}  fit {res['fit_seconds']:.1f}s  "
            f"rss {res['peak_rss_mb']:.0f} MB  -> {path.name}",
            flush=True,
        )
        return
    folds = [int(f) for f in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]
    for fold in folds:
        for lib in LIBS:
            subprocess.run([sys.executable, __file__, "run", lib, str(fold)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
