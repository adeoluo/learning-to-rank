"""Write a /rank request body from one MSLR-WEB10K test query (used by docs/demo/serve.tape).

    python scripts/demo_query.py <mslr dir> <out.json>

Takes the first fold 1 test query with at least three relevance grades, picks two documents per
grade in shuffled order, and labels each id with its true grade so the ranking can be read.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from learning_to_rank.datasets import load_mslr_fold


def main() -> None:
    data_dir, out = sys.argv[1], sys.argv[2]
    _, _, test = load_mslr_fold(data_dir, 1, max_queries=50)
    rng = np.random.default_rng(0)
    for q in range(test.n_queries):
        lo, hi = int(test.offsets[q]), int(test.offsets[q + 1])
        y = test.y[lo:hi]
        grades = np.unique(y)
        if grades.shape[0] >= 3:
            break
    picks = np.concatenate([np.flatnonzero(y == g)[:2] for g in grades[-3:]])
    picks = rng.permutation(picks)
    body = {
        "documents": test.X[lo:hi][picks].round(4).tolist(),
        "ids": [f"doc{int(i)}-grade{int(y[i])}" for i in picks],
        "top_k": 6,
    }
    with open(out, "w") as fh:
        json.dump(body, fh)


if __name__ == "__main__":
    main()
