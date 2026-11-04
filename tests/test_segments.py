from __future__ import annotations

import numpy as np
import pytest

from learning_to_rank.segments import frequency_segments, length_segments, segment_report


def test_frequency_segments_sizes() -> None:
    f = np.arange(100, dtype=float)
    seg = frequency_segments(f)
    assert (
        (seg == "head").sum() == 10 and (seg == "torso").sum() == 40 and (seg == "tail").sum() == 50
    )
    assert seg[99] == "head" and seg[0] == "tail"
    ties = frequency_segments(np.zeros(10), shares=(0.5, 0.5), names=("a", "b"))
    assert ties.tolist() == ["a"] * 5 + ["b"] * 5
    with pytest.raises(ValueError):
        frequency_segments(f, shares=(0.5, 0.6))


def test_length_segments() -> None:
    assert length_segments(np.array([0, 1, 4, 5, 9])).tolist() == ["unknown", "1", "4", "5+", "5+"]


def test_segment_report() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(size=400)
    better = base + np.where(np.arange(400) < 200, 0.2, 0.0)
    seg = np.where(np.arange(400) < 200, "head", "tail")
    rows = segment_report({"base": base, "new": better}, seg, baseline="base", n_permutations=500)
    assert [r.segment for r in rows] == ["head", "tail"]
    assert rows[0].diffs["new"] == pytest.approx(0.2)
    assert rows[0].adjusted_p_values["new"] < 0.01
    assert rows[1].diffs["new"] == pytest.approx(0.0)
    assert rows[1].p_values["new"] == pytest.approx(1.0)
    assert rows[0].to_dict()["n_queries"] == 200
    ordered = segment_report(
        {"base": base, "new": better},
        seg,
        baseline="base",
        order=["tail", "head", "x"],
        n_permutations=10,
    )
    assert ordered[2].n_queries == 0 and np.isnan(ordered[2].means["base"])
    with pytest.raises(ValueError):
        segment_report({"base": base}, seg, baseline="other")
    with pytest.raises(ValueError):
        segment_report({"base": base[:5]}, seg, baseline="base")
