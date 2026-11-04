"""Neural rankers: an MLP scoring function trained with pointwise, pairwise or listwise losses.

Losses (all computed on padded query batches with a validity mask):

* ``pointwise``: squared error between score and label.
* ``ranknet``: pairwise logistic loss over document pairs with different labels (Burges et al.,
  2005), averaged over pairs.
* ``lambdarank``: RankNet pairs weighted by ``|delta NDCG|`` of swapping the pair in the current
  ranking (the neural form of LambdaRank).
* ``listnet``: cross-entropy between ``softmax(labels)`` and ``softmax(scores)`` (Cao et al.,
  2007, top-one probability).
* ``softmax``: cross-entropy with target distribution ``labels / sum(labels)``. With labels that
  are inverse-propensity-weighted clicks this is the unbiased listwise loss used for learning from
  click logs (Ai et al., 2018); queries whose labels sum to zero are skipped.

Features are transformed with ``sign(x) * log1p(|x|)`` and standardized with training-set
statistics, the usual preprocessing for MSLR-style count features (Qin et al., 2021).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.data import RankingData
from learning_to_rank.metrics import evaluate

Loss = Literal["pointwise", "ranknet", "lambdarank", "listnet", "softmax"]
LOSSES: tuple[str, ...] = ("pointwise", "ranknet", "lambdarank", "listnet", "softmax")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("neural rankers need PyTorch: uv sync --extra neural") from exc
    return torch


@dataclass(frozen=True)
class NeuralParams:
    loss: Loss = "listnet"
    hidden: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30
    batch_docs: int = 8192
    max_pairs_per_batch: int = 4_000_000
    early_stopping_rounds: int = 5
    eval_at: int = 10
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if not self.hidden or min(self.hidden) < 1:
            raise ValueError("hidden must list positive layer widths")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.epochs < 1:
            raise ValueError("invalid dropout, learning_rate or epochs")
        if self.batch_docs < 1 or self.max_pairs_per_batch < 1 or self.early_stopping_rounds < 0:
            raise ValueError("invalid batch_docs, max_pairs_per_batch or early_stopping_rounds")


def resolve_device(device: str) -> str:
    """``auto`` picks Apple MPS when available, else CPU."""

    if device != "auto":
        return device
    torch = _torch()
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class FeatureScaler:
    """``sign(x) log1p(|x|)`` followed by standardization (constant features map to 0)."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    @staticmethod
    def _log(X: NDArray[np.floating]) -> NDArray[np.float32]:
        Xf = np.asarray(X, dtype=np.float32)
        out: NDArray[np.float32] = (np.sign(Xf) * np.log1p(np.abs(Xf))).astype(np.float32)
        return out

    @classmethod
    def fit(cls, X: NDArray[np.floating]) -> FeatureScaler:
        L = cls._log(X)
        mean = L.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = L.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.float32]:
        out: NDArray[np.float32] = ((self._log(X) - self.mean) / self.std).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def query_batches(
    offsets: NDArray[np.int64],
    batch_docs: int,
    max_pairs: int,
    rng: np.random.Generator | None,
) -> Iterator[NDArray[np.int64]]:
    """Batches of query positions with similar list lengths.

    Queries are sorted by length (with random tie order when ``rng`` is given) and packed until
    the padded batch would exceed ``batch_docs`` documents or ``max_pairs`` document pairs; the
    batch order is shuffled when ``rng`` is given.
    """

    sizes = np.diff(offsets)
    keys = rng.random(sizes.shape[0]) if rng is not None else np.arange(sizes.shape[0])
    order = np.lexsort((keys, sizes))
    batches: list[NDArray[np.int64]] = []
    start = 0
    while start < order.shape[0]:
        end = start + 1
        while end < order.shape[0]:
            width = int(sizes[order[end]])
            count = end + 1 - start
            if count * width > batch_docs or count * width * width > max_pairs:
                break
            end += 1
        batches.append(order[start:end].astype(np.int64))
        start = end
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    yield from batches


def _pad(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    offsets: NDArray[np.int64],
    queries: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    sizes = offsets[queries + 1] - offsets[queries]
    width = int(np.max(sizes))
    B = queries.shape[0]
    Xb = np.zeros((B, width, X.shape[1]), dtype=np.float32)
    yb = np.zeros((B, width), dtype=np.float32)
    mask = np.zeros((B, width), dtype=bool)
    qi = np.repeat(np.arange(B), sizes)
    pos = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    rows = np.repeat(offsets[queries] - (np.cumsum(sizes) - sizes), sizes) + np.arange(
        int(sizes.sum())
    )
    Xb[qi, pos] = X[rows]
    yb[qi, pos] = y[rows]
    mask[qi, pos] = True
    return Xb, yb, mask


def ranking_loss(loss: Loss, scores: Any, labels: Any, mask: Any) -> Any:
    """Mean loss of one padded batch (``scores``, ``labels``, ``mask`` are ``[B, L]``)."""

    torch = _torch()
    m = mask.to(scores.dtype)
    if loss == "pointwise":
        return (((scores - labels) ** 2) * m).sum() / m.sum().clamp(min=1.0)
    if loss in ("ranknet", "lambdarank"):
        diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        better = (labels.unsqueeze(2) > labels.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)
        weight = better.to(scores.dtype)
        if loss == "lambdarank":
            with torch.no_grad():
                weight = weight * _delta_ndcg(scores, labels, mask)
        pair_loss = torch.nn.functional.softplus(-diff)
        return (pair_loss * weight).sum() / better.sum().clamp(min=1)
    neg = torch.finfo(scores.dtype).min
    log_p = torch.log_softmax(scores.masked_fill(~mask, neg), dim=1)
    if loss == "listnet":
        target = torch.softmax(labels.masked_fill(~mask, neg), dim=1)
        keep = mask.any(dim=1)
    else:
        total = (labels * m).sum(dim=1, keepdim=True)
        keep = total.squeeze(1) > 0
        target = (labels * m) / total.clamp(min=1e-12)
    per_query = -(target * log_p.masked_fill(~mask, 0.0)).sum(dim=1)
    return (per_query * keep).sum() / keep.sum().clamp(min=1)


def _delta_ndcg(scores: Any, labels: Any, mask: Any) -> Any:
    """``|delta NDCG|`` of swapping each pair in the ranking induced by ``scores``."""

    torch = _torch()
    neg = torch.finfo(scores.dtype).min
    order = torch.argsort(scores.masked_fill(~mask, neg), dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    disc = 1.0 / torch.log2(ranks.to(scores.dtype) + 2.0)
    gains = (torch.exp2(labels) - 1.0) * mask
    ideal = torch.sort(gains, dim=1, descending=True).values
    idcg = (
        ideal
        / torch.log2(torch.arange(ideal.shape[1], device=ideal.device, dtype=scores.dtype) + 2.0)
    ).sum(1)
    inv = torch.where(idcg > 0, 1.0 / idcg.clamp(min=1e-12), torch.zeros_like(idcg))
    dg = gains.unsqueeze(2) - gains.unsqueeze(1)
    dd = disc.unsqueeze(2) - disc.unsqueeze(1)
    return (dg * dd).abs() * inv.view(-1, 1, 1)


@dataclass
class NeuralResult:
    valid_ndcg: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    fit_seconds: float = 0.0


class NeuralRanker:
    """An MLP ranker; ``fit`` on :class:`RankingData`, ``predict`` on raw feature matrices."""

    def __init__(self, params: NeuralParams | None = None, **overrides: Any) -> None:
        base = params or NeuralParams()
        self.params = NeuralParams(**{**asdict(base), **overrides}) if overrides else base
        self.scaler: FeatureScaler | None = None
        self.model: Any = None
        self.result = NeuralResult()

    def _build(self, n_features: int) -> Any:
        torch = _torch()
        layers: list[Any] = []
        width = n_features
        for h in self.params.hidden:
            layers += [
                torch.nn.Linear(width, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.params.dropout),
            ]
            width = h
        layers.append(torch.nn.Linear(width, 1))
        return torch.nn.Sequential(*layers)

    def fit(
        self,
        train: RankingData,
        valid: RankingData | None = None,
        *,
        callback: Callable[[int, float, float | None], None] | None = None,
    ) -> NeuralRanker:
        torch = _torch()
        p = self.params
        t0 = time.perf_counter()
        torch.manual_seed(p.seed)
        rng = np.random.default_rng(p.seed)
        self.scaler = FeatureScaler.fit(train.X)
        Xtr = self.scaler.transform(train.X)
        device = torch.device(resolve_device(p.device))
        self.model = self._build(train.n_features).to(device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )
        best, best_state, stale = -np.inf, None, 0
        result = NeuralResult()
        for epoch in range(1, p.epochs + 1):
            self.model.train()
            losses = []
            for queries in query_batches(train.offsets, p.batch_docs, p.max_pairs_per_batch, rng):
                Xb, yb, mb = _pad(Xtr, train.y, train.offsets, queries)
                xb = torch.from_numpy(Xb).to(device)
                lb = torch.from_numpy(yb).to(device)
                mk = torch.from_numpy(mb).to(device)
                scores = self.model(xb).squeeze(-1)
                loss = ranking_loss(p.loss, scores, lb, mk)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            result.train_loss.append(float(np.mean(losses)) if losses else float("nan"))
            score = None
            if valid is not None:
                score = evaluate(valid, self.predict(valid.X), ks=(p.eval_at,))[f"ndcg@{p.eval_at}"]
                result.valid_ndcg.append(score)
                if score > best:
                    best, stale, result.best_epoch = score, 0, epoch
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    stale += 1
            if callback is not None:
                callback(epoch, result.train_loss[-1], score)
            if valid is not None and p.early_stopping_rounds and stale >= p.early_stopping_rounds:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif valid is None:
            result.best_epoch = len(result.train_loss)
        result.fit_seconds = time.perf_counter() - t0
        self.result = result
        return self

    def predict(
        self, X: NDArray[np.floating] | RankingData, batch: int = 65536
    ) -> NDArray[np.float64]:
        torch = _torch()
        if self.model is None or self.scaler is None:
            raise RuntimeError("the model is not fitted")
        feats = X.X if isinstance(X, RankingData) else np.asarray(X)
        Xs = self.scaler.transform(feats)
        device = next(self.model.parameters()).device
        self.model.eval()
        out = np.empty(Xs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for s in range(0, Xs.shape[0], batch):
                xb = torch.from_numpy(Xs[s : s + batch]).to(device)
                out[s : s + batch] = self.model(xb).squeeze(-1).cpu().double().numpy()
        return out

    def save(self, path: str | Path) -> None:
        torch = _torch()
        if self.model is None or self.scaler is None:
            raise RuntimeError("the model is not fitted")
        params = asdict(self.params)
        params["hidden"] = list(params["hidden"])
        torch.save(
            {
                "format": "learning-to-rank/neural",
                "params": params,
                "mean": torch.from_numpy(self.scaler.mean),
                "std": torch.from_numpy(self.scaler.std),
                "n_features": int(self.scaler.mean.shape[0]),
                "state": self.model.state_dict(),
                "result": asdict(self.result),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralRanker:
        torch = _torch()
        # weights_only: tensors and plain containers only, so a model file cannot run code.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "learning-to-rank/neural":
            raise ValueError("not a learning-to-rank neural model file")
        params = dict(payload["params"])
        params["hidden"] = tuple(params["hidden"])
        params["device"] = "cpu"
        model = cls(NeuralParams(**params))
        model.scaler = FeatureScaler(payload["mean"].numpy(), payload["std"].numpy())
        model.model = model._build(int(payload["n_features"]))
        model.model.load_state_dict(payload["state"])
        model.result = NeuralResult(**payload["result"])
        return model
