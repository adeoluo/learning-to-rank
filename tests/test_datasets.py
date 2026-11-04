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


def _write_istella(path: Path, rng: np.random.Generator) -> None:
    import io
    import tarfile

    from learning_to_rank.data import synthetic, to_svmlight
    from learning_to_rank.datasets import ISTELLA_FEATURES, ISTELLA_SPLITS

    with tarfile.open(path, "w:gz") as tar:
        for i, member in enumerate(ISTELLA_SPLITS.values()):
            d = synthetic(20, (3, 9), ISTELLA_FEATURES, seed=i)
            raw = to_svmlight(d, precision=7).encode()
            info = tarfile.TarInfo(member)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def test_prepare_and_load_istella(tmp_path: Path) -> None:
    from learning_to_rank import LambdaMART, evaluate
    from learning_to_rank.datasets import ISTELLA_FEATURES, load_istella, prepare_istella

    archive = tmp_path / "istella.tar.gz"
    _write_istella(archive, np.random.default_rng(0))
    stats = prepare_istella(archive, tmp_path / "out", sample_every=2, lines_per_chunk=50)
    assert stats["train"]["queries"] == 20 and stats["features"] == ISTELLA_FEATURES
    train, test = load_istella(tmp_path / "out", "train"), load_istella(tmp_path / "out", "test")
    assert train.bins.dtype == np.uint8 and train.bins.shape == (
        stats["train"]["rows"],
        ISTELLA_FEATURES,
    )
    m = LambdaMART(n_estimators=5, min_data_in_leaf=2).fit_binned(
        train, load_istella(tmp_path / "out", "valid")
    )
    scores = m.predict(test.bins.astype(np.float32))
    data = RankingData(
        test.bins.astype(np.float32),
        test.y,
        np.repeat(np.arange(len(test.offsets) - 1), np.diff(test.offsets)),
    )
    assert 0 <= evaluate(data, scores)["ndcg@10"] <= 1
    assert len(m.result.valid_ndcg) == 5
