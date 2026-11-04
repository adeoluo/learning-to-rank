from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning_to_rank import RankingData, evaluate
from learning_to_rank.neural import (
    LOSSES,
    FeatureScaler,
    NeuralParams,
    NeuralRanker,
    query_batches,
    ranking_loss,
    resolve_device,
)


@pytest.fixture(autouse=True, scope="module")
def _torch_required() -> None:
    pytest.importorskip("torch")


@pytest.mark.parametrize("loss", LOSSES)
def test_every_loss_learns(loss: str, medium: tuple[RankingData, ...]) -> None:
    train, valid, test = medium
    m = NeuralRanker(
        loss=loss, epochs=6, hidden=(32,), batch_docs=1024, learning_rate=3e-3, seed=1
    ).fit(train, valid)
    ours = evaluate(test, m.predict(test))["ndcg@10"]
    rand = evaluate(test, np.random.default_rng(0).normal(size=test.n_docs))["ndcg@10"]
    assert ours > rand + 0.15, (loss, ours, rand)
    assert 1 <= m.result.best_epoch <= 6 and len(m.result.train_loss) >= 1


def test_early_stopping_and_callback(medium: tuple[RankingData, ...]) -> None:
    train, valid, _ = medium
    seen: list[int] = []
    m = NeuralRanker(epochs=30, early_stopping_rounds=1, learning_rate=0.05, hidden=(16,)).fit(
        train, valid, callback=lambda e, loss, v: seen.append(e)
    )
    assert len(seen) < 30 and seen == list(range(1, len(seen) + 1))
    assert m.result.best_epoch == int(np.argmax(m.result.valid_ndcg)) + 1


def test_without_validation(small: RankingData) -> None:
    m = NeuralRanker(epochs=2, hidden=(8,)).fit(small)
    assert m.result.best_epoch == 2 and m.result.valid_ndcg == []


def test_save_load(tmp_path: Path, small: RankingData) -> None:
    import torch

    m = NeuralRanker(epochs=1, hidden=(8, 4), loss="ranknet").fit(small)
    path = tmp_path / "nn.pt"
    m.save(path)
    loaded = NeuralRanker.load(path)
    np.testing.assert_allclose(loaded.predict(small.X), m.predict(small.X), rtol=1e-6)
    assert loaded.params.hidden == (8, 4)
    torch.save({"format": "other"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        NeuralRanker.load(tmp_path / "bad.pt")


def test_unfitted_errors() -> None:
    m = NeuralRanker()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 3)))
    with pytest.raises(RuntimeError):
        m.save("x.pt")


def test_param_validation() -> None:
    for bad in (
        {"loss": "hinge"},
        {"hidden": ()},
        {"dropout": 1.0},
        {"epochs": 0},
        {"batch_docs": 0},
    ):
        with pytest.raises(ValueError):
            NeuralParams(**bad)  # type: ignore[arg-type]
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "mps")


def test_scaler_handles_constants_and_nan() -> None:
    X = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 1.0]], dtype=np.float32)
    sc = FeatureScaler.fit(np.nan_to_num(X))
    out = sc.transform(X)
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 1], 0.0)


def test_batches_cover_every_query_once() -> None:
    sizes = np.array([3, 50, 7, 7, 120, 1, 30])
    offsets = np.r_[0, np.cumsum(sizes)]
    seen = np.concatenate(list(query_batches(offsets, 100, 10_000, np.random.default_rng(0))))
    assert sorted(seen.tolist()) == list(range(len(sizes)))
    for b in query_batches(offsets, 100, 10_000, None):
        w = sizes[b].max()
        assert len(b) == 1 or (len(b) * w <= 100 and len(b) * w * w <= 10_000)


def test_loss_values_on_small_cases() -> None:
    import torch

    s = torch.tensor([[2.0, 0.0, 9.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    rn = ranking_loss("ranknet", s, y, mask)
    assert float(rn) == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(-2.0))))
    pw = ranking_loss("pointwise", s, y, mask)
    assert float(pw) == pytest.approx(((2 - 1) ** 2 + 0) / 2)
    sm = ranking_loss("softmax", s, y, mask)
    assert float(sm) == pytest.approx(float(-torch.log_softmax(torch.tensor([2.0, 0.0]), 0)[0]))
    ln = ranking_loss("listnet", s, y, mask)
    target = torch.softmax(torch.tensor([1.0, 0.0]), 0)
    expected = -(target * torch.log_softmax(torch.tensor([2.0, 0.0]), 0)).sum()
    assert float(ln) == pytest.approx(float(expected))
    lr = ranking_loss("lambdarank", s, y, mask)
    assert float(lr) > 0
    zero = ranking_loss("softmax", s, torch.zeros_like(y), mask)
    assert float(zero) == 0.0
