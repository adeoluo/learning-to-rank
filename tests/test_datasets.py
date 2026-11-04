from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning_to_rank.data import RankingData
from learning_to_rank.datasets import (
    MSLR_FEATURES,
    fold_partitions,
    load_mslr_fold,
    mslr_partition,
    mslr_query_clicks,
    mslr_query_length,
    prepare_mslr,
)


def test_partitions_and_folds() -> None:
    assert mslr_partition(np.array([1, 4, 7, 10, 13, 16, 29998])).tolist() == [1, 2, 3, 4, 5, 1, 5]
    with pytest.raises(ValueError):
        mslr_partition(np.array([2]))
    assert fold_partitions(1) == ((1, 2, 3), 4, 5)
    assert fold_partitions(2) == ((2, 3, 4), 5, 1)
    assert fold_partitions(5) == ((5, 1, 2), 3, 4)
    for f in range(1, 6):
        tr, va, te = fold_partitions(f)
        assert sorted([*tr, va, te]) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        fold_partitions(6)


def _write_parquet(path: Path, qids: list[int], rng: np.random.Generator) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sizes = rng.integers(2, 5, size=len(qids))
    labels = [rng.integers(0, 5, size=s).astype(float).tolist() for s in sizes]
    feats = [rng.normal(size=(s, MSLR_FEATURES)).tolist() for s in sizes]
    table = pa.table({"query": qids, "labels": labels, "features": feats, "n": sizes.tolist()})
    pq.write_table(table, path, row_group_size=3)


def test_prepare_and_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir()
    _write_parquet(src / "hf_train.parquet", [1, 4, 7, 16, 19, 22], rng)
    _write_parquet(src / "hf_validation.parquet", [10, 25], rng)
    _write_parquet(src / "hf_test.parquet", [13, 28], rng)
    stats = prepare_mslr(src, tmp_path / "out")
    assert stats["queries"] == 10
    assert sum(stats[f"S{s}_rows"] for s in range(1, 6)) == stats["rows"]
    tr, va, te = load_mslr_fold(tmp_path / "out", 1)
    assert set(tr.query_ids % 15) == {1, 4, 7}
    assert set(va.query_ids % 15) == {10} and set(te.query_ids % 15) == {13}
    assert tr.n_features == MSLR_FEATURES
    _tr2, va2, te2 = load_mslr_fold(tmp_path / "out", 2)
    assert set(te2.query_ids % 15) == {1} and set(va2.query_ids % 15) == {13}
    small, _, _ = load_mslr_fold(tmp_path / "out", 1, max_queries=2)
    assert small.n_queries == 2


def test_query_length_and_clicks() -> None:
    X = np.zeros((4, MSLR_FEATURES), dtype=np.float32)
    X[0, 4], X[0, 9] = 2, 0.5  # 2 of 4 terms covered
    X[1, 4], X[1, 9] = 1, 0.25
    X[:, 133] = [3, 1, 0, 5]
    d = RankingData.from_arrays(X, [0, 1, 0, 1], [1, 1, 2, 2])
    assert mslr_query_length(d).tolist() == [4, 0]
    assert mslr_query_clicks(d).tolist() == [4.0, 5.0]
