"""LambdaMART on Istella-S: this package against LightGBM and XGBoost on identical binned data.

Istella-S (3,408,630 documents, 220 features) is converted once by `prepare_istella` into
255-bin uint8 matrices (bin edges fitted on every tenth training row), because float32 features
would need 3 GB. All three libraries train on these bin indices, so they see exactly the same
information; LightGBM reads them through `lgb.Sequence` and XGBoost through a `DataIter`, in
batches, to stay within memory. Same budget as on MSLR: 500 trees, learning rate 0.1, 31 leaves,
8 threads. Each library runs in its own process.

    uv run --group validation python benchmarks/istella.py prepare
    uv run --group validation python benchmarks/istella.py all

Results: benchmarks/results/istella_<lib>.json
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
from common import DATA, ROOT, Timer, machine, peak_rss_mb, write

TREES, LR, LEAVES, MIN_DATA, THREADS, BATCH = 500, 0.1, 31, 20, 8, 100_000
OUT = DATA / "istella-s"


def batches(bins: np.ndarray) -> Any:
    for s in range(0, bins.shape[0], BATCH):
        yield s, bins[s : s + BATCH].astype(np.float32)


def run(lib: str) -> dict[str, Any]:
    from learning_to_rank.datasets import load_istella
    from learning_to_rank.metrics import per_query

    train, test = load_istella(OUT, "train"), load_istella(OUT, "test")
    scores = np.empty(test.n_docs)
    construct = 0.0
    if lib == "ours":
        from learning_to_rank import LambdaMART

        model = LambdaMART(
            n_estimators=TREES, learning_rate=LR, num_leaves=LEAVES, min_data_in_leaf=MIN_DATA
        )
        with Timer() as t:
            model.fit_binned(train)
        with Timer() as tp:
            for s, X in batches(test.bins):
                scores[s : s + X.shape[0]] = model.predict(X)
        version = "this package"
    elif lib == "lightgbm":
        import lightgbm as lgb

        class Rows(lgb.Sequence):  # type: ignore[misc]
            batch_size = BATCH

            def __init__(self, bins: np.ndarray) -> None:
                self.bins = bins

            def __getitem__(self, idx: Any) -> np.ndarray:
                return np.asarray(self.bins[idx], dtype=np.float64)

            def __len__(self) -> int:
                return int(self.bins.shape[0])

        params = {
            "objective": "lambdarank",
            "learning_rate": LR,
            "num_leaves": LEAVES,
            "min_data_in_leaf": MIN_DATA,
            "max_bin": 255,
            "num_threads": THREADS,
            "verbose": -1,
            "seed": 0,
        }
        with Timer() as t:
            with Timer() as tc:
                ds = lgb.Dataset(
                    Rows(train.bins), train.y, group=np.diff(train.offsets), params=params
                ).construct()
            booster = lgb.train(params, ds, TREES)
        construct = tc.seconds
        with Timer() as tp:
            for s, X in batches(test.bins):
                scores[s : s + X.shape[0]] = booster.predict(X, num_threads=THREADS)
        version = lgb.__version__
    elif lib == "xgboost":
        import xgboost as xgb

        qidx = np.repeat(np.arange(train.offsets.shape[0] - 1), np.diff(train.offsets))

        class It(xgb.DataIter):  # type: ignore[misc]
            def __init__(self) -> None:
                self.pos = 0
                super().__init__()

            def next(self, input_data: Any) -> bool:
                if self.pos >= train.n_docs:
                    return False
                e = min(self.pos + BATCH, train.n_docs)
                input_data(
                    data=train.bins[self.pos : e].astype(np.float32),
                    label=train.y[self.pos : e],
                    qid=qidx[self.pos : e],
                )
                self.pos = e
                return True

            def reset(self) -> None:
                self.pos = 0

        params = {
            "objective": "rank:ndcg",
            "eta": LR,
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_leaves": LEAVES,
            "max_depth": 0,
            "max_bin": 256,
            "nthread": THREADS,
            "seed": 0,
        }
        with Timer() as t:
            with Timer() as tc:
                dtrain = xgb.QuantileDMatrix(It(), max_bin=256)
            booster = xgb.train(params, dtrain, TREES)
        construct = tc.seconds
        with Timer() as tp:
            for s, X in batches(test.bins):
                scores[s : s + X.shape[0]] = booster.predict(xgb.DMatrix(X, nthread=THREADS))
        version = xgb.__version__
    else:
        raise SystemExit(f"unknown library {lib}")

    pq = per_query(test.y, scores, test.offsets)
    m = {k: float(v.mean()) for k, v in pq.items()}
    return {
        "library": lib,
        "version": version,
        "train_docs": train.n_docs,
        "train_queries": int(train.offsets.shape[0] - 1),
        "test_docs": test.n_docs,
        "test_queries": int(test.offsets.shape[0] - 1),
        "trees": TREES,
        "fit_seconds": t.seconds,
        "construct_seconds": construct,
        "predict_seconds": tp.seconds,
        "peak_rss_mb": peak_rss_mb(),
        "metrics": m,
        "per_query_ndcg@10": pq["ndcg@10"].round(6).tolist(),
        "machine": machine(),
    }


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prepare":
        from learning_to_rank.datasets import prepare_istella

        with Timer() as t:
            stats = prepare_istella(DATA / "istella-s-letor.tar.gz", OUT)
        print(stats, f"{t.seconds:.0f}s")
        return
    if cmd == "run":
        res = run(sys.argv[2])
        write(f"istella_{sys.argv[2]}.json", res)
        print(
            f"{sys.argv[2]:<9} ndcg@10 {res['metrics']['ndcg@10']:.4f}  "
            f"fit {res['fit_seconds']:.0f}s  rss {res['peak_rss_mb']:.0f} MB",
            flush=True,
        )
        return
    for lib in ("ours", "lightgbm", "xgboost"):
        subprocess.run([sys.executable, __file__, "run", lib], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
