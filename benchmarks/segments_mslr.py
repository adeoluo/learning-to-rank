"""Segment analysis and significance over all 10,000 MSLR-WEB10K queries.

Every query is a test query in exactly one of the five folds, so concatenating the per-query
test NDCG@10 of the five folds covers each query once. Queries are segmented by

* popularity: MSLR has no query frequencies, so we use the total logged query-url clicks of the
  query's documents (feature 134) as the popularity signal; head = top 10 percent of queries,
  torso = next 40 percent, tail = bottom 50 percent;
* query length in terms, recovered from features 5 and 10 (covered terms / covered ratio);

and every system is compared with LightGBM per segment with Fisher's randomization test
(10,000 sign flips), Holm-adjusted across segments. Needs the results of gbdt_mslr.py and
neural_mslr.py.

    uv run python benchmarks/segments_mslr.py

Results: benchmarks/results/segments_mslr.json
"""

from __future__ import annotations

import glob
import json
from typing import Any

import numpy as np
from common import MSLR, RESULTS, machine, write

SYSTEMS = {
    "LambdaMART (this package)": "gbdt_mslr_ours_fold{}.json",
    "LightGBM": "gbdt_mslr_lightgbm_fold{}.json",
    "XGBoost": "gbdt_mslr_xgboost_fold{}.json",
    "MLP, ListNet": "neural_mslr_listnet_fold{}.json",
    "MLP, RankNet": "neural_mslr_ranknet_fold{}.json",
}


def per_query_all(pattern: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for fold in range(1, 6):
        path = RESULTS / pattern.format(fold)
        d = json.loads(path.read_text())
        out.update(zip(d["per_query"]["qid"], d["per_query"]["ndcg@10"], strict=True))
    return out


def main() -> None:
    from learning_to_rank.data import group_offsets
    from learning_to_rank.datasets import MSLR_QUERY_URL_CLICKS, query_length_from_columns
    from learning_to_rank.segments import frequency_segments, length_segments, segment_report
    from learning_to_rank.significance import compare

    X = np.load(MSLR / "X.npy", mmap_mode="r")
    qid = np.load(MSLR / "qid.npy").astype(np.int64)
    offsets = group_offsets(qid)
    qids = qid[offsets[:-1]]
    covered, ratio = np.asarray(X[:, 4]), np.asarray(X[:, 9])
    clicks = np.asarray(X[:, MSLR_QUERY_URL_CLICKS - 1], dtype=np.float64)
    length = query_length_from_columns(covered, ratio, offsets)
    popularity = np.add.reduceat(clicks, offsets[:-1])

    available = {
        name: pat
        for name, pat in SYSTEMS.items()
        if len(glob.glob(str(RESULTS / pat.format("*")))) == 5
    }
    values = {}
    for name, pat in available.items():
        pq = per_query_all(pat)
        values[name] = np.array([pq[int(q)] for q in qids])
    base = "LightGBM"
    overall: dict[str, Any] = {}
    for name, v in values.items():
        if name == base:
            continue
        overall[name] = compare(v, values[base], metric="ndcg@10").to_dict()
    pop = frequency_segments(popularity)
    lens = length_segments(length)
    report: dict[str, Any] = {
        "n_queries": int(qids.shape[0]),
        "systems": list(values),
        "baseline": base,
        "overall_vs_baseline": overall,
        "popularity_thresholds": {
            seg: [float(popularity[pop == seg].min()), float(popularity[pop == seg].max())]
            for seg in ("head", "torso", "tail")
        },
        "popularity": [
            r.to_dict()
            for r in segment_report(values, pop, baseline=base, order=["head", "torso", "tail"])
        ],
        "query_length": [
            r.to_dict()
            for r in segment_report(
                values, lens, baseline=base, order=["1", "2", "3", "4", "5+", "unknown"]
            )
        ],
        "machine": machine(),
    }
    write("segments_mslr.json", report)
    for row in report["popularity"] + report["query_length"]:
        print(row["segment"], row["n_queries"], {k: round(v, 4) for k, v in row["means"].items()})


if __name__ == "__main__":
    main()
