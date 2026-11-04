"""Neural rankers on MSLR-WEB10K, five folds: pointwise, RankNet, LambdaRank and ListNet losses.

Same MLP (136 -> 256 -> 256 -> 128 -> 1, ReLU, dropout 0.1), Adam at 1e-3, batches of about
8,192 documents, at most 30 epochs with early stopping on validation NDCG@10 (patience 5),
seed 0, Apple MPS. One process per run.

    uv run --extra neural python benchmarks/neural_mslr.py all
    uv run --extra neural python benchmarks/neural_mslr.py run listnet 1

Results: benchmarks/results/neural_mslr_<loss>_fold<k>.json
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from common import MSLR, ROOT, Timer, machine, peak_rss_mb, write

LOSSES = ("pointwise", "ranknet", "lambdarank", "listnet")


def run(loss: str, fold: int, device: str) -> dict[str, Any]:
    from learning_to_rank.datasets import load_mslr_fold
    from learning_to_rank.metrics import evaluate, per_query
    from learning_to_rank.neural import NeuralRanker

    train, valid, test = load_mslr_fold(MSLR, fold)
    model = NeuralRanker(loss=loss, epochs=30, early_stopping_rounds=5, seed=0, device=device)  # type: ignore[arg-type]
    with Timer() as t:
        model.fit(train, valid)
    scores = model.predict(test)
    pq = per_query(test.y, scores, test.offsets)
    return {
        "loss": loss,
        "fold": fold,
        "device": device,
        "epochs_run": len(model.result.train_loss),
        "best_epoch": model.result.best_epoch,
        "valid_ndcg@10": model.result.valid_ndcg,
        "fit_seconds": t.seconds,
        "peak_rss_mb": peak_rss_mb(),
        "metrics": evaluate(test, scores),
        "per_query": {"qid": test.query_ids.tolist(), "ndcg@10": pq["ndcg@10"].round(6).tolist()},
        "machine": machine(),
    }


def main() -> None:
    device = "mps"
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        loss, fold = sys.argv[2], int(sys.argv[3])
        res = run(loss, fold, device)
        write(f"neural_mslr_{loss}_fold{fold}.json", res)
        print(
            f"{loss:<10} fold {fold}  ndcg@10 {res['metrics']['ndcg@10']:.4f}  "
            f"epochs {res['epochs_run']} (best {res['best_epoch']})  fit {res['fit_seconds']:.0f}s",
            flush=True,
        )
        return
    folds = [int(f) for f in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]
    for fold in folds:
        for loss in LOSSES:
            subprocess.run([sys.executable, __file__, "run", loss, str(fold)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
