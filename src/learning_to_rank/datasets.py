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

Istella-S (Lucchese et al., 2016) has 33,018 queries and 3,408,630 documents with 220 features
and grades 0 to 4, split 60/20/20 into train, validation and test; it is distributed for
non-commercial research under the Istella LETOR licence and is converted by
``prepare_istella``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.data import RankingData

if TYPE_CHECKING:
    from learning_to_rank.gbdt import BinnedData

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
    y = np.load(d / "y.npy")
    qid = np.load(d / "qid.npy")
    part = np.load(d / "part.npy")
    header = np.load(d / "X.npy", mmap_mode="r")
    offset, width = int(header.offset), int(header.shape[1])
    del header
    train_parts, valid_part, test_part = fold_partitions(fold)

    def take(parts: tuple[int, ...]) -> RankingData:
        # Partitions are contiguous row ranges (prepare_mslr sorts by partition), so each one
        # is read straight from the file without mapping the whole matrix into memory.
        ranges = []
        left = max_queries
        for p in parts:
            idx = np.flatnonzero(part == p)
            start, stop = int(idx[0]), int(idx[-1]) + 1
            if stop - start != idx.shape[0]:
                raise ValueError("partitions must be contiguous; rerun prepare_mslr")
            if left is not None:
                if left <= 0:
                    break
                q = qid[start:stop]
                starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
                if starts.shape[0] > left:
                    stop = start + int(starts[left])
                left -= min(left, int(starts.shape[0]))
            ranges.append((start, stop))
        n = sum(b - a for a, b in ranges)
        X = np.empty((n, width), dtype=np.float32)
        pos = 0
        with open(d / "X.npy", "rb") as fh:
            for a, b in ranges:
                fh.seek(offset + a * width * 4)
                X[pos : pos + b - a] = np.fromfile(
                    fh, dtype=np.float32, count=(b - a) * width
                ).reshape(-1, width)
                pos += b - a
        rows = np.concatenate([np.arange(a, b) for a, b in ranges])
        return RankingData(X, y[rows].astype(np.float32), qid[rows])

    return take(train_parts), take((valid_part,)), take((test_part,))


def mslr_query_length(data: RankingData) -> NDArray[np.int64]:
    """Number of query terms per query, recovered from MSLR features 5 and 10.

    Feature 5 counts the query terms a document covers and feature 10 is that count divided by
    the query length, so length = round(f5 / f10) for any document with f10 > 0. Queries whose
    documents cover no term get 0 (unknown).
    """

    return query_length_from_columns(
        data.X[:, MSLR_COVERED_TERMS_WHOLE - 1],
        data.X[:, MSLR_COVERED_RATIO_WHOLE - 1],
        data.offsets,
    )


def query_length_from_columns(
    covered: NDArray[np.floating], ratio: NDArray[np.floating], offsets: NDArray[np.int64]
) -> NDArray[np.int64]:
    """``mslr_query_length`` from the two feature columns and query offsets."""

    c = np.asarray(covered, dtype=np.float64)
    r = np.asarray(ratio, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.where(r > 0, np.rint(c / np.where(r > 0, r, 1.0)), 0.0)
    nq = offsets.shape[0] - 1
    qi = np.repeat(np.arange(nq), np.diff(offsets))
    out = np.zeros(nq, dtype=np.float64)
    np.maximum.at(out, qi, est)
    return out.astype(np.int64)


def mslr_query_clicks(data: RankingData) -> NDArray[np.float64]:
    """Total query-url click count (feature 134) over each query's documents.

    MSLR does not ship query frequencies; this sum of logged clicks is the closest popularity
    signal in the data and is used to split queries into head, torso and tail.
    """

    clicks = data.X[:, MSLR_QUERY_URL_CLICKS - 1].astype(np.float64)
    return np.add.reduceat(clicks, data.offsets[:-1]) if data.n_docs else np.zeros(0)


ISTELLA_FEATURES = 220
ISTELLA_SPLITS = {
    "train": "sample/train.txt",
    "valid": "sample/vali.txt",
    "test": "sample/test.txt",
}


def _istella_chunks(archive: Path, member: str, lines_per_chunk: int) -> Iterator[RankingData]:
    """Parse one split of the Istella-S archive in chunks, streaming from the tar.gz."""

    import tarfile

    from learning_to_rank.data import parse_svmlight

    with tarfile.open(archive, "r:gz") as tar:
        fh = tar.extractfile(member)
        if fh is None:
            raise ValueError(f"{member} not found in {archive}")
        buf: list[bytes] = []
        for line in fh:
            buf.append(line)
            if len(buf) >= lines_per_chunk:
                yield parse_svmlight(b"".join(buf).decode("ascii"), ISTELLA_FEATURES)
                buf = []
        if buf:
            yield parse_svmlight(b"".join(buf).decode("ascii"), ISTELLA_FEATURES)


def prepare_istella(
    archive: str | Path,
    out_dir: str | Path,
    *,
    sample_every: int = 10,
    max_bin: int = 255,
    lines_per_chunk: int = 200_000,
) -> dict[str, Any]:
    """Convert Istella-S (3,408,630 documents x 220 features) into 255-bin ``uint8`` matrices.

    Float32 features would take 3 GB, so the archive is streamed twice: the first pass keeps
    every ``sample_every``-th training row to fit the bin edges (as LightGBM fits its bins on a
    row sample), the second writes bin indices, labels and query ids for each split. Every
    library in the benchmark then trains on the same binned matrices.
    """

    from learning_to_rank.binning import BinMapper

    src, out = Path(archive), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = np.concatenate(
        [
            c.X[::sample_every]
            for c in _istella_chunks(src, ISTELLA_SPLITS["train"], lines_per_chunk)
        ]
    )
    mapper = BinMapper.fit(sample, max_bin=max_bin, sample_size=sample.shape[0])
    del sample
    stats: dict[str, Any] = {"features": ISTELLA_FEATURES, "max_bin": max_bin}
    for split, member in ISTELLA_SPLITS.items():
        rows = 0
        labels: list[NDArray[np.float32]] = []
        qids: list[NDArray[np.int64]] = []
        with open(out / f"{split}.bins", "wb") as fh:
            for chunk in _istella_chunks(src, member, lines_per_chunk):
                fh.write(mapper.transform(chunk.X).tobytes())
                labels.append(chunk.y)
                qids.append(chunk.qid)
                rows += chunk.n_docs
        q = np.concatenate(qids)
        np.save(out / f"{split}_y.npy", np.concatenate(labels).astype(np.int8))
        np.save(out / f"{split}_qid.npy", q)
        stats[split] = {"rows": rows, "queries": int(np.unique(q).shape[0])}
    (out / "edges.json").write_text(json.dumps(mapper.to_lists()))
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_istella(data_dir: str | Path, split: str) -> BinnedData:
    """One Istella-S split from a ``prepare_istella`` directory (bins memory-mapped)."""

    from learning_to_rank.data import group_offsets
    from learning_to_rank.gbdt import BinnedData

    d = Path(data_dir)
    y = np.load(d / f"{split}_y.npy").astype(np.float32)
    qid = np.load(d / f"{split}_qid.npy")
    bins = np.memmap(
        d / f"{split}.bins", dtype=np.uint8, mode="r", shape=(y.shape[0], ISTELLA_FEATURES)
    )
    return BinnedData(np.asarray(bins), y, group_offsets(qid))
