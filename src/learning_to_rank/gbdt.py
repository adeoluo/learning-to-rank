"""LambdaMART: gradient boosted regression trees trained with LambdaRank gradients.

The default backend is the Rust core (histogram trees grown leaf-wise, parallel histogram
building and gradient computation); ``backend="python"`` runs the numpy reference in
:mod:`learning_to_rank.gbdt_reference`, which produces the same trees on small data.
The defaults follow LightGBM's ``lambdarank`` objective (31 leaves, learning rate 0.1,
``min_data_in_leaf`` 20, truncation level 30, normalized lambdas, 255 bins).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from learning_to_rank.backend import extension, resolve_backend
from learning_to_rank.binning import BinMapper
from learning_to_rank.data import RankingData
from learning_to_rank.gbdt_reference import (
    SplitMix64,
    feature_mask_reference,
    grow_tree_reference,
    ideal_dcg_reference,
    lambdarank_gradients_reference,
)

Objective = Literal["lambdarank", "regression"]


@dataclass(frozen=True)
class GBDTParams:
    """Boosting hyperparameters (names follow LightGBM where the meaning is the same)."""

    objective: Objective = "lambdarank"
    n_estimators: int = 500
    learning_rate: float = 0.1
    num_leaves: int = 31
    max_depth: int = 0
    min_data_in_leaf: int = 20
    min_sum_hessian_in_leaf: float = 1e-3
    lambda_l2: float = 0.0
    min_gain_to_split: float = 0.0
    feature_fraction: float = 1.0
    max_bin: int = 255
    bin_sample_size: int = 200_000
    truncation_level: int = 30
    sigma: float = 1.0
    norm: bool = True
    eval_at: int = 10
    early_stopping_rounds: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.objective not in ("lambdarank", "regression"):
            raise ValueError("objective must be 'lambdarank' or 'regression'")
        if self.n_estimators < 1 or self.num_leaves < 1:
            raise ValueError("n_estimators and num_leaves must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 2 <= self.max_bin <= 256:
            raise ValueError("max_bin must be in [2, 256]")
        if min(self.max_depth, self.min_data_in_leaf, self.truncation_level) < 0:
            raise ValueError("max_depth, min_data_in_leaf and truncation_level must be >= 0")
        if self.lambda_l2 < 0 or self.min_sum_hessian_in_leaf < 0 or self.sigma <= 0:
            raise ValueError("lambda_l2 and min_sum_hessian_in_leaf must be >= 0, sigma > 0")
        if self.eval_at < 1 or self.early_stopping_rounds < 0:
            raise ValueError("eval_at must be positive and early_stopping_rounds >= 0")


_ARRAYS: dict[str, Any] = {
    "node_offsets": np.int64,
    "leaf_offsets": np.int64,
    "split_feature": np.int32,
    "threshold_bin": np.int32,
    "threshold": np.float32,
    "left": np.int32,
    "right": np.int32,
    "split_gain": np.float64,
    "leaf_value": np.float64,
    "leaf_count": np.int64,
}


@dataclass
class Ensemble:
    """A tree ensemble as flat arrays; tree ``t`` owns nodes ``node_offsets[t]:[t+1]`` and
    leaves ``leaf_offsets[t]:[t+1]``. Child pointers ``< 0`` are leaves (``~child``)."""

    n_features: int
    node_offsets: NDArray[np.int64]
    leaf_offsets: NDArray[np.int64]
    split_feature: NDArray[np.int32]
    threshold_bin: NDArray[np.int32]
    threshold: NDArray[np.float32]
    left: NDArray[np.int32]
    right: NDArray[np.int32]
    split_gain: NDArray[np.float64]
    leaf_value: NDArray[np.float64]
    leaf_count: NDArray[np.int64]

    @classmethod
    def empty(cls, n_features: int) -> Ensemble:
        arrays = {
            k: np.zeros(1 if k.endswith("offsets") else 0, dtype=t) for k, t in _ARRAYS.items()
        }
        return cls(n_features=n_features, **arrays)

    @classmethod
    def from_bytes(cls, n_features: int, raw: dict[str, bytes]) -> Ensemble:
        arrays = {k: np.frombuffer(raw[k], dtype=t).copy() for k, t in _ARRAYS.items()}
        return cls(n_features=n_features, **arrays)

    @property
    def n_trees(self) -> int:
        return int(self.node_offsets.shape[0] - 1)

    def truncate(self, k: int) -> Ensemble:
        k = max(0, min(k, self.n_trees))
        nn, nl = int(self.node_offsets[k]), int(self.leaf_offsets[k])
        return Ensemble(
            n_features=self.n_features,
            node_offsets=self.node_offsets[: k + 1].copy(),
            leaf_offsets=self.leaf_offsets[: k + 1].copy(),
            split_feature=self.split_feature[:nn].copy(),
            threshold_bin=self.threshold_bin[:nn].copy(),
            threshold=self.threshold[:nn].copy(),
            left=self.left[:nn].copy(),
            right=self.right[:nn].copy(),
            split_gain=self.split_gain[:nn].copy(),
            leaf_value=self.leaf_value[:nl].copy(),
            leaf_count=self.leaf_count[:nl].copy(),
        )

    def predict(self, X: NDArray[np.floating], backend: str = "auto") -> NDArray[np.float64]:
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        if Xc.ndim != 2 or Xc.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got shape {Xc.shape}")
        if Xc.shape[0] == 0:
            return np.zeros(0)
        if resolve_backend(backend) == "rust":
            raw = extension().predict(
                Xc,
                self.n_features,
                self.node_offsets,
                self.split_feature,
                self.threshold,
                self.left,
                self.right,
                self.leaf_offsets,
                self.leaf_value,
            )
            return np.frombuffer(raw, dtype=np.float64).copy()
        return self._predict_reference(Xc)

    def _predict_reference(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        n = X.shape[0]
        out = np.zeros(n)
        rows = np.arange(n)
        for t in range(self.n_trees):
            a, b = int(self.node_offsets[t]), int(self.node_offsets[t + 1])
            la = int(self.leaf_offsets[t])
            if a == b:
                out += self.leaf_value[la]
                continue
            node = np.zeros(n, dtype=np.int64)
            active = np.ones(n, dtype=bool)
            leaf = np.zeros(n, dtype=np.int64)
            while active.any():
                r = rows[active]
                nd = node[r] + a
                x = X[r, self.split_feature[nd]]
                go_left = np.isnan(x) | (x <= self.threshold[nd])
                child = np.where(go_left, self.left[nd], self.right[nd]).astype(np.int64)
                done = child < 0
                leaf[r[done]] = ~child[done]
                node[r[~done]] = child[~done]
                active[r[done]] = False
            out += self.leaf_value[la + leaf]
        return out

    def feature_importance(self, kind: Literal["gain", "split"] = "gain") -> NDArray[np.float64]:
        if kind not in ("gain", "split"):
            raise ValueError("kind must be 'gain' or 'split'")
        weights = self.split_gain if kind == "gain" else None
        return np.bincount(self.split_feature, weights=weights, minlength=self.n_features).astype(
            np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_features": self.n_features}
        for k in _ARRAYS:
            out[k] = getattr(self, k).tolist()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Ensemble:
        arrays = {k: np.asarray(d[k], dtype=t) for k, t in _ARRAYS.items()}
        ens = cls(n_features=int(d["n_features"]), **arrays)
        ens._validate()
        return ens

    def _validate(self) -> None:
        no, lo = self.node_offsets, self.leaf_offsets
        if no.shape != lo.shape or no.shape[0] < 1 or no[0] != 0 or lo[0] != 0:
            raise ValueError("invalid tree offsets")
        if no[-1] != self.split_feature.shape[0] or lo[-1] != self.leaf_value.shape[0]:
            raise ValueError("offsets do not match array sizes")
        if np.any(np.diff(lo) != np.diff(no) + 1):
            raise ValueError("each tree must have one more leaf than internal nodes")
        if self.split_feature.size and (
            int(np.min(self.split_feature)) < 0
            or int(np.max(self.split_feature)) >= self.n_features
        ):
            raise ValueError("split feature out of range")
        for t in range(self.n_trees):
            a, b = int(no[t]), int(no[t + 1])
            n_nodes, n_leaves = b - a, int(lo[t + 1] - lo[t])
            for i in range(a, b):
                for c in (int(self.left[i]), int(self.right[i])):
                    if c >= 0 and not (i - a) < c < n_nodes:
                        raise ValueError("child pointers must point forward")
                    if c < 0 and ~c >= n_leaves:
                        raise ValueError("leaf index out of range")


class _ReferenceBooster:
    """Same interface as the Rust ``Booster``, implemented with the numpy reference."""

    def __init__(
        self,
        bins: NDArray[np.uint8],
        train: RankingData,
        edges: list[NDArray[np.float32]],
        p: GBDTParams,
        valid_bins: NDArray[np.uint8] | None,
        valid: RankingData | None,
    ) -> None:
        self.bins, self.train, self.edges, self.p = bins, train, edges, p
        self.n_bins = [int(e.shape[0]) + 1 for e in edges]
        self.valid_bins, self.valid = valid_bins, valid
        self.scores = np.zeros(train.n_docs)
        self.valid_scores = np.zeros(valid.n_docs if valid is not None else 0)
        self.trees: list[Any] = []
        self.rng = SplitMix64(p.seed)
        self.tree_params: dict[str, float] = {
            "num_leaves": p.num_leaves,
            "max_depth": p.max_depth,
            "min_data_in_leaf": p.min_data_in_leaf,
            "min_sum_hessian_in_leaf": p.min_sum_hessian_in_leaf,
            "lambda_l2": p.lambda_l2,
            "min_gain_to_split": p.min_gain_to_split,
        }

    @staticmethod
    def _ndcg(
        y: NDArray[np.float32], scores: NDArray[np.float64], offsets: NDArray[np.int64], k: int
    ) -> float:
        vals = []
        for q in range(offsets.shape[0] - 1):
            s, e = int(offsets[q]), int(offsets[q + 1])
            idcg = ideal_dcg_reference(y[s:e], k)
            if idcg <= 0:
                vals.append(1.0)
                continue
            order = np.argsort(-scores[s:e], kind="stable")[:k]
            yy = y[s:e][order].astype(np.float64)
            dcg = float(np.cumsum((np.exp2(yy) - 1.0) / np.log2(np.arange(yy.shape[0]) + 2.0))[-1])
            vals.append(dcg / idcg)
        return float(np.mean(vals)) if vals else 0.0

    def step(self, k: int) -> list[float | None]:
        out: list[float | None] = []
        y = self.train.y.astype(np.float64)
        for _ in range(k):
            if self.p.objective == "lambdarank":
                g, h = lambdarank_gradients_reference(
                    y,
                    self.scores,
                    self.train.offsets,
                    self.p.truncation_level,
                    self.p.sigma,
                    self.p.norm,
                )
            else:
                g, h = self.scores - y, np.ones_like(y)
            mask = feature_mask_reference(self.train.n_features, self.p.feature_fraction, self.rng)
            idx = np.arange(self.train.n_docs, dtype=np.int64)
            tree, ranges = grow_tree_reference(
                self.bins,
                self.n_bins,
                self.edges,
                g,
                h,
                idx,
                mask,
                self.tree_params,
                self.p.learning_rate,
            )
            for leaf, (s, e) in enumerate(ranges):
                self.scores[idx[s:e]] += tree.leaf_value[leaf]
            self.trees.append(tree)
            if self.valid is not None and self.valid_bins is not None:
                self.valid_scores += _predict_binned_tree(tree, self.valid_bins)
                out.append(
                    self._ndcg(self.valid.y, self.valid_scores, self.valid.offsets, self.p.eval_at)
                )
            else:
                out.append(None)
        return out

    def train_ndcg(self) -> float:
        return self._ndcg(self.train.y, self.scores, self.train.offsets, self.p.eval_at)

    def train_scores(self) -> bytes:
        return self.scores.tobytes()

    def truncate(self, k: int) -> None:
        self.trees = self.trees[:k]

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    def export(self) -> dict[str, bytes]:
        cols: dict[str, list[Any]] = {k: [] for k in _ARRAYS}
        cols["node_offsets"].append(0)
        cols["leaf_offsets"].append(0)
        for t in self.trees:
            for k in (
                "split_feature",
                "threshold_bin",
                "threshold",
                "left",
                "right",
                "split_gain",
                "leaf_value",
                "leaf_count",
            ):
                cols[k].extend(getattr(t, k))
            cols["node_offsets"].append(len(cols["split_feature"]))
            cols["leaf_offsets"].append(len(cols["leaf_value"]))
        return {k: np.asarray(v, dtype=_ARRAYS[k]).tobytes() for k, v in cols.items()}


def _predict_binned_tree(tree: Any, bins: NDArray[np.uint8]) -> NDArray[np.float64]:
    n = bins.shape[0]
    if not tree.split_feature:
        return np.full(n, tree.leaf_value[0])
    out = np.zeros(n)
    for r in range(n):
        node = 0
        while True:
            f = tree.split_feature[node]
            child = tree.left[node] if bins[r, f] <= tree.threshold_bin[node] else tree.right[node]
            if child < 0:
                out[r] = tree.leaf_value[~child]
                break
            node = child
    return out


@dataclass
class FitResult:
    """What happened during ``fit``: validation NDCG per tree, best iteration, timings."""

    valid_ndcg: list[float] = field(default_factory=list)
    best_iteration: int = 0
    best_score: float = float("nan")
    train_ndcg: float = float("nan")
    fit_seconds: float = 0.0
    binning_seconds: float = 0.0


class LambdaMART:
    """Gradient boosted trees for ranking; ``fit`` on :class:`RankingData`, ``predict`` on
    raw feature matrices."""

    def __init__(
        self, params: GBDTParams | None = None, *, backend: str = "auto", **overrides: Any
    ) -> None:
        base = params or GBDTParams()
        known = {f.name for f in fields(GBDTParams)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"unknown parameters: {sorted(unknown)}")
        self.params = replace(base, **overrides) if overrides else base
        self.backend = resolve_backend(backend)
        self.ensemble: Ensemble | None = None
        self.result = FitResult()

    def fit(
        self,
        train: RankingData,
        valid: RankingData | None = None,
        *,
        callback: Callable[[int, float | None], None] | None = None,
        chunk: int = 10,
    ) -> LambdaMART:
        """Train; with ``valid`` and ``early_stopping_rounds > 0`` keep the best iteration."""

        p = self.params
        if valid is not None and valid.n_features != train.n_features:
            raise ValueError("train and validation data have different feature counts")
        t0 = time.perf_counter()
        mapper = BinMapper.fit(
            train.X,
            max_bin=p.max_bin,
            sample_size=p.bin_sample_size,
            seed=p.seed,
            backend=self.backend,
        )
        bins = mapper.transform(train.X, backend=self.backend)
        vbins = mapper.transform(valid.X, backend=self.backend) if valid is not None else None
        t1 = time.perf_counter()
        booster: Any
        if self.backend == "rust":
            kwargs: dict[str, Any] = {}
            if valid is not None:
                kwargs = {
                    "valid_bins": vbins,
                    "valid_labels": valid.y,
                    "valid_offsets": valid.offsets,
                }
            booster = extension().Booster(
                bins,
                train.n_features,
                train.y,
                train.offsets,
                mapper.to_lists(),
                objective=p.objective,
                learning_rate=p.learning_rate,
                num_leaves=p.num_leaves,
                max_depth=p.max_depth,
                min_data_in_leaf=p.min_data_in_leaf,
                min_sum_hessian_in_leaf=p.min_sum_hessian_in_leaf,
                lambda_l2=p.lambda_l2,
                min_gain_to_split=p.min_gain_to_split,
                feature_fraction=p.feature_fraction,
                truncation_level=p.truncation_level,
                sigma=p.sigma,
                norm=p.norm,
                eval_at=p.eval_at,
                seed=p.seed,
                **kwargs,
            )
        else:
            booster = _ReferenceBooster(bins, train, mapper.edges, p, vbins, valid)
        history: list[float] = []
        best_it, best = 0, -np.inf
        done = 0
        while done < p.n_estimators:
            k = min(chunk, p.n_estimators - done)
            for v in booster.step(k):
                done += 1
                if callback is not None:
                    callback(done, v)
                if v is not None:
                    history.append(v)
                    if v > best:
                        best, best_it = v, done
            if (
                p.early_stopping_rounds
                and valid is not None
                and done - best_it >= p.early_stopping_rounds
            ):
                break
        keep = best_it if (p.early_stopping_rounds and valid is not None and best_it > 0) else done
        booster.truncate(keep)
        t2 = time.perf_counter()
        self.ensemble = Ensemble.from_bytes(train.n_features, booster.export())
        self.result = FitResult(
            valid_ndcg=history,
            best_iteration=keep,
            best_score=float(best) if history else float("nan"),
            train_ndcg=float(booster.train_ndcg()) if keep == done else float("nan"),
            fit_seconds=t2 - t0,
            binning_seconds=t1 - t0,
        )
        return self

    def _require(self) -> Ensemble:
        if self.ensemble is None:
            raise RuntimeError("the model is not fitted")
        return self.ensemble

    def predict(self, X: NDArray[np.floating] | RankingData) -> NDArray[np.float64]:
        feats = X.X if isinstance(X, RankingData) else X
        return self._require().predict(feats, backend=self.backend)

    def feature_importance(self, kind: Literal["gain", "split"] = "gain") -> NDArray[np.float64]:
        return self._require().feature_importance(kind)

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "learning-to-rank/lambdamart",
            "version": 1,
            "params": asdict(self.params),
            "result": asdict(self.result),
            "ensemble": self._require().to_dict(),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "auto") -> LambdaMART:
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != "learning-to-rank/lambdamart":
            raise ValueError("not a learning-to-rank LambdaMART model file")
        model = cls(GBDTParams(**payload["params"]), backend=backend)
        model.ensemble = Ensemble.from_dict(payload["ensemble"])
        model.result = FitResult(**payload.get("result", {}))
        return model
