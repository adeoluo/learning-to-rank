from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from conftest import BACKENDS
from learning_to_rank.binning import BinMapper, fit_edges_reference


def test_few_values_get_own_bins() -> None:
    e = fit_edges_reference(np.array([3.0, 1.0, 2.0, 1.0]), 255)
    assert e.tolist() == [1.0, 2.0]
    assert fit_edges_reference(np.array([5.0, 5.0]), 8).size == 0
    assert fit_edges_reference(np.array([np.nan, 1.0]), 8).size == 0
    with pytest.raises(ValueError):
        fit_edges_reference(np.zeros(3), 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transform_matches_raw_threshold(backend: str) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    X[:, 2] = np.round(X[:, 2])
    X[5, 1] = np.nan
    m = BinMapper.fit(X, max_bin=32, backend=backend)
    b = m.transform(X, backend=backend)
    assert b.dtype == np.uint8 and b.shape == X.shape
    assert b[5, 1] == 0
    for j in range(4):
        for t in range(len(m.edges[j])):
            col = X[:, j]
            ok = ~np.isnan(col)
            np.testing.assert_array_equal(b[ok, j] <= t, col[ok] <= m.edges[j][t])
    assert max(m.n_bins) <= 32


@pytest.mark.parametrize("backend", BACKENDS)
def test_validation(backend: str) -> None:
    X = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        BinMapper.fit(X, max_bin=1, backend=backend)
    m = BinMapper.fit(X, backend=backend)
    with pytest.raises(ValueError):
        m.transform(np.zeros((2, 3)), backend=backend)
    assert BinMapper.from_lists(m.to_lists()).n_bins == [1, 1]


def test_sampling_is_seeded() -> None:
    X = np.random.default_rng(3).normal(size=(5000, 3)).astype(np.float32)
    a = BinMapper.fit(X, sample_size=1000, seed=1, backend="python")
    b = BinMapper.fit(X, sample_size=1000, seed=1, backend="python")
    assert all(np.array_equal(x, y) for x, y in zip(a.edges, b.edges, strict=True))


@settings(max_examples=50, deadline=None)
@given(
    X=hnp.arrays(
        np.float32,
        st.tuples(st.integers(1, 300), st.integers(1, 4)),
        elements=st.one_of(
            st.integers(-20, 20).map(float),
            st.floats(-1e3, 1e3, width=32),
            st.just(float("nan")),
        ),
    ),
    max_bin=st.integers(2, 64),
)
def test_rust_and_reference_agree(X: np.ndarray, max_bin: int) -> None:
    ref = BinMapper.fit(X, max_bin=max_bin, backend="python")
    for backend in BACKENDS:
        got = BinMapper.fit(X, max_bin=max_bin, backend=backend)
        for a, b in zip(ref.edges, got.edges, strict=True):
            np.testing.assert_array_equal(a, b)
            assert len(a) <= max_bin - 1
        np.testing.assert_array_equal(ref.transform(X, "python"), got.transform(X, backend))
