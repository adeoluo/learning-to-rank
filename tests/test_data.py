from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import BACKENDS
from learning_to_rank.data import (
    RankingData,
    group_offsets,
    parse_svmlight,
    read_svmlight,
    synthetic,
    to_svmlight,
)

TEXT = """2 qid:10 1:0.5 3:2 # doc a
0 qid:10 2:1.5

1 qid:11 1:-0.001
"""


def test_offsets_and_properties() -> None:
    d = RankingData.from_arrays(np.zeros((5, 2)), [0, 1, 2, 0, 1], [7, 7, 3, 3, 3])
    assert d.offsets.tolist() == [0, 2, 5]
    assert d.n_queries == 2 and d.n_docs == 5 and d.n_features == 2
    assert d.query_ids.tolist() == [7, 3]
    assert d.group_sizes.tolist() == [2, 3]
    assert d.query_index().tolist() == [0, 0, 1, 1, 1]
    assert group_offsets(np.zeros(0, dtype=np.int64)).tolist() == [0]


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        RankingData.from_arrays(np.zeros((3, 1)), [0, 0, 0], [1, 2, 1])
    with pytest.raises(ValueError, match="2-D"):
        RankingData(
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="same number"):
        RankingData.from_arrays(np.zeros((3, 1)), [0, 0], [1, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        RankingData.from_arrays(np.zeros((1, 1)), [np.nan], [1])


def test_sort_groups_rows() -> None:
    d = RankingData.from_arrays([[1.0], [2.0], [3.0]], [0, 1, 2], [2, 1, 2], sort=True)
    assert d.qid.tolist() == [1, 2, 2]
    assert d.X[:, 0].tolist() == [2.0, 1.0, 3.0]


def test_subset_concat_and_labels(small: RankingData) -> None:
    sub = small.subset([3, 0])
    assert sub.query_ids.tolist() == [small.query_ids[3], small.query_ids[0]]
    s, e = small.offsets[3], small.offsets[4]
    np.testing.assert_array_equal(sub.X[: e - s], small.X[s:e])
    both = RankingData.concat([small.subset([0]), small.subset([1])])
    assert both.n_queries == 2
    relabeled = small.with_labels(np.ones(small.n_docs))
    assert relabeled.y.sum() == small.n_docs
    with pytest.raises(IndexError):
        small.subset([small.n_queries])
    with pytest.raises(ValueError):
        RankingData.concat([])
    assert small.subset([]).n_docs == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_parse_svmlight(backend: str) -> None:
    d = parse_svmlight(TEXT, backend=backend)
    assert d.n_features == 3
    assert d.y.tolist() == [2.0, 0.0, 1.0]
    assert d.qid.tolist() == [10, 10, 11]
    np.testing.assert_allclose(d.X, [[0.5, 0, 2], [0, 1.5, 0], [-0.001, 0, 0]], rtol=1e-7)
    wide = parse_svmlight("1 qid:1 2:3\n", n_features=5, backend=backend)
    assert wide.X.shape == (1, 5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "bad",
    ["x qid:1 1:2\n", "1 1:2\n", "1 qid:1 0:2\n", "1 qid:1 1:z\n", "1 qid:1 12\n", "1 qid:a 1:1\n"],
)
def test_parse_errors(backend: str, bad: str) -> None:
    with pytest.raises(ValueError):
        parse_svmlight(bad, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_width_overflow(backend: str) -> None:
    with pytest.raises(ValueError):
        parse_svmlight("1 qid:1 5:3\n", n_features=4, backend=backend)


def test_round_trip_and_files(tmp_path: Path, small: RankingData) -> None:
    text = to_svmlight(small, precision=9)
    plain = tmp_path / "a.txt"
    plain.write_text(text)
    zipped = tmp_path / "a.txt.gz"
    with gzip.open(zipped, "wt") as fh:
        fh.write(text)
    for p in (plain, zipped):
        d = read_svmlight(p)
        np.testing.assert_array_equal(d.X, small.X)
        np.testing.assert_array_equal(d.y, small.y)
        np.testing.assert_array_equal(d.qid, small.qid)


@settings(max_examples=40, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, 4),
            st.integers(0, 50),
            st.dictionaries(st.integers(1, 12), st.floats(-1e6, 1e6, width=32), max_size=6),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_parsers_agree(rows: list[tuple[int, int, dict[int, float]]]) -> None:
    lines = []
    for label, qid, feats in rows:
        body = " ".join(f"{k}:{v!r}" for k, v in sorted(feats.items()))
        lines.append(f"{label} qid:{qid} {body}".rstrip())
    text = "\n".join(lines) + "\n"
    ref = parse_svmlight(text, n_features=12, backend="python") if _contiguous(rows) else None
    if ref is None:
        return
    for backend in BACKENDS:
        got = parse_svmlight(text, n_features=12, backend=backend)
        np.testing.assert_array_equal(got.X, ref.X)
        np.testing.assert_array_equal(got.y, ref.y)


def _contiguous(rows: list[tuple[int, int, dict[int, float]]]) -> bool:
    seen: list[int] = []
    for _, q, _ in rows:
        if not seen or seen[-1] != q:
            if q in seen:
                return False
            seen.append(q)
    return True


def test_synthetic_is_reproducible() -> None:
    a, b = synthetic(30, seed=4), synthetic(30, seed=4)
    np.testing.assert_array_equal(a.X, b.X)
    assert set(np.unique(a.y)) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert (a.y == 0).mean() > 0.4
