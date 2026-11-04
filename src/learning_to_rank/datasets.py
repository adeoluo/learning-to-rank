"""Public learning-to-rank datasets: MSLR-WEB10K (five folds) and Istella-S.

MSLR-WEB10K (Qin and Liu, 2013) has 10,000 queries and 1,200,192 judged query-document pairs
with 136 features and relevance grades 0 to 4. Microsoft publishes it as five partitions S1..S5
arranged into five folds (three partitions train, one validates, one tests). The official
OneDrive link no longer serves anonymous downloads, so ``scripts/download_data.sh`` fetches the
Hugging Face parquet mirror (``philipphager/MSLR-WEB10k``) of Fold 1. Every MSLR query id is
congruent to 1, 4, 7, 10 or 13 modulo 15, and these five classes are exactly S1..S5: the
Fold 1 validation file (S4) holds only ids = 10 mod 15, the test file (S5) only 13 mod 15, and
the training file (S1 to S3) holds 1, 4 and 7 with 2,000 queries each. ``prepare_mslr`` stores
all rows once with their partition, and ``load_mslr_fold`` assembles any of the five folds
from them (the Fold 1 row counts reproduce the published 723,412 / 235,259 / 241,521).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.data import RankingData

MSLR_FEATURES = 136
MSLR_PARTITION_OF_RESIDUE = {1: 1, 4: 2, 7: 3, 10: 4, 13: 5}
MSLR_SPLITS = ("hf_train.parquet", "hf_validation.parquet", "hf_test.parquet")

# 1-based MSLR feature ids used by the segment analysis (from the MSLR feature list).
MSLR_COVERED_TERMS_WHOLE = 5  # covered query term number, whole document
MSLR_COVERED_RATIO_WHOLE = 10  # covered query term ratio, whole document
MSLR_QUERY_URL_CLICKS = 134  # query-url click count


def mslr_partition(qid: NDArray[np.integer]) -> NDArray[np.int8]:
    """Partition S1..S5 (as 1..5) of each MSLR-WEB10K query id; raises for foreign ids."""

    residue = np.asarray(qid, dtype=np.int64) % 15
    part = np.zeros(residue.shape, dtype=np.int8)
    for r, s in MSLR_PARTITION_OF_RESIDUE.items():
        part[residue == r] = s
    if (part == 0).any():
        raise ValueError("query ids outside the MSLR-WEB10K residue classes")
    return part


def fold_partitions(fold: int) -> tuple[tuple[int, int, int], int, int]:
    """(train partitions, validation partition, test partition) of fold 1..5."""

    if fold not in range(1, 6):
        raise ValueError("fold must be in 1..5")

    def s(i: int) -> int:
        return (fold - 1 + i) % 5 + 1

    return (s(0), s(1), s(2)), s(3), s(4)


def prepare_mslr(source_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Convert the parquet mirror into ``X.npy`` (float32), ``y.npy``, ``qid.npy``, ``part.npy``.

    Rows are written grouped by partition, then by query id; reads one row group at a time.
    """

    import pyarrow.parquet as pq

    src, out = Path(source_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for name in MSLR_SPLITS:
        n_rows += int(pq.read_table(src / name, columns=["n"]).column("n").to_numpy().sum())
    X = np.lib.format.open_memmap(
        out / "X.raw.npy", mode="w+", dtype=np.float32, shape=(n_rows, MSLR_FEATURES)
    )
    y = np.empty(n_rows, dtype=np.int8)
    qid = np.empty(n_rows, dtype=np.int32)
    pos = 0
    for name in MSLR_SPLITS:
        pf = pq.ParquetFile(src / name)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g)
            queries = t.column("query").to_numpy()
            sizes = t.column("n").to_numpy()
            labels = t.column("labels").combine_chunks().flatten().to_numpy()
            feats = t.column("features").combine_chunks().flatten().flatten().to_numpy()
            k = int(sizes.sum())
            if labels.shape[0] != k or feats.shape[0] != k * MSLR_FEATURES:
                raise ValueError(f"{name}: row group {g} has inconsistent sizes")
            X[pos : pos + k] = feats.reshape(k, MSLR_FEATURES).astype(np.float32)
            y[pos : pos + k] = labels.astype(np.int8)
            qid[pos : pos + k] = np.repeat(queries, sizes).astype(np.int32)
            pos += k
    if pos != n_rows:
        raise ValueError("row count mismatch while converting")
    part = mslr_partition(qid)
    order = np.lexsort((qid, part))
    np.save(out / "y.npy", y[order])
    np.save(out / "qid.npy", qid[order])
    np.save(out / "part.npy", part[order])
    Xs = np.lib.format.open_memmap(
        out / "X.npy", mode="w+", dtype=np.float32, shape=(n_rows, MSLR_FEATURES)
    )
    step = 100_000
    for s in range(0, n_rows, step):
        Xs[s : s + step] = X[order[s : s + step]]
    Xs.flush()
    del X, Xs
    (out / "X.raw.npy").unlink()
    stats = {
        "rows": n_rows,
        "queries": int(np.unique(qid).shape[0]),
        **{f"S{s}_rows": int((part == s).sum()) for s in range(1, 6)},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_mslr_fold(
    data_dir: str | Path, fold: int = 1, *, max_queries: int | None = None
) -> tuple[RankingData, RankingData, RankingData]:
    """(train, validation, test) of one MSLR-WEB10K fold from a ``prepare_mslr`` directory.

    ``max_queries`` keeps the first queries of each split (for quick runs and demos).
    """

    d = Path(data_dir)
    X = np.load(d / "X.npy", mmap_mode="r")
    y = np.load(d / "y.npy")
    qid = np.load(d / "qid.npy")
    part = np.load(d / "part.npy")
    train_parts, valid_part, test_part = fold_partitions(fold)

    def take(parts: tuple[int, ...]) -> RankingData:
        rows = np.concatenate([np.flatnonzero(part == p) for p in parts])
        if max_queries is not None:
            q = qid[rows]
            starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
            if starts.shape[0] > max_queries:
                rows = rows[: starts[max_queries]]
        return RankingData(np.asarray(X[rows]), y[rows].astype(np.float32), qid[rows])

    return take(train_parts), take((valid_part,)), take((test_part,))


def mslr_query_length(data: RankingData) -> NDArray[np.int64]:
    """Number of query terms per query, recovered from MSLR features 5 and 10.

    Feature 5 counts the query terms a document covers and feature 10 is that count divided by
    the query length, so length = round(f5 / f10) for any document with f10 > 0. Queries whose
    documents cover no term get 0 (unknown).
    """

    covered = data.X[:, MSLR_COVERED_TERMS_WHOLE - 1].astype(np.float64)
    ratio = data.X[:, MSLR_COVERED_RATIO_WHOLE - 1].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.where(ratio > 0, np.rint(covered / ratio), 0.0)
    qi = data.query_index()
    out = np.zeros(data.n_queries, dtype=np.float64)
    np.maximum.at(out, qi, est)
    return out.astype(np.int64)


def mslr_query_clicks(data: RankingData) -> NDArray[np.float64]:
    """Total query-url click count (feature 134) over each query's documents.

    MSLR does not ship query frequencies; this sum of logged clicks is the closest popularity
    signal in the data and is used to split queries into head, torso and tail.
    """

    clicks = data.X[:, MSLR_QUERY_URL_CLICKS - 1].astype(np.float64)
    return np.add.reduceat(clicks, data.offsets[:-1]) if data.n_docs else np.zeros(0)
