"""Command line interface: ``python -m learning_to_rank <command>``.

Commands:

* ``info``: version, backend and thread count.
* ``prepare-mslr``: convert the MSLR-WEB10K parquet mirror into numpy arrays (all five folds).
* ``train``: fit LambdaMART (or a neural ranker) on an MSLR fold or LETOR files.
* ``evaluate``: NDCG@k, MAP, MRR and ERR of a saved model.
* ``compare``: two saved models on the same queries with a paired t-test and a randomization test.
* ``clicks``: simulate a click log, fit click models and estimate position bias.
* ``serve``: the HTTP ranking service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from learning_to_rank import __version__
from learning_to_rank.backend import extension, rust_available
from learning_to_rank.data import RankingData, read_svmlight
from learning_to_rank.gbdt import GBDTParams, LambdaMART
from learning_to_rank.metrics import evaluate, per_query

METRIC_COLUMNS = ("ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "map", "mrr", "err@10")


def _add_data_args(p: argparse.ArgumentParser, *, split: bool = True) -> None:
    g = p.add_argument_group("data (an MSLR directory from prepare-mslr, or LETOR files)")
    g.add_argument("--data", type=Path, help="directory written by prepare-mslr")
    g.add_argument("--fold", type=int, default=1, help="MSLR fold 1..5 (default 1)")
    g.add_argument(
        "--max-queries", type=int, default=None, help="keep the first N queries per split"
    )
    if split:
        g.add_argument("--split", choices=("train", "valid", "test"), default="test")
        g.add_argument("--file", type=Path, help="a LETOR file instead of --data")


def _mslr(args: argparse.Namespace) -> tuple[RankingData, RankingData, RankingData]:
    from learning_to_rank.datasets import load_mslr_fold

    if args.data is None:
        raise SystemExit("give --data (a prepare-mslr directory) or LETOR files")
    return load_mslr_fold(args.data, args.fold, max_queries=args.max_queries)


def _split(args: argparse.Namespace) -> RankingData:
    if getattr(args, "file", None) is not None:
        return read_svmlight(args.file)
    tr, va, te = _mslr(args)
    return {"train": tr, "valid": va, "test": te}[args.split]


def load_model(path: Path) -> tuple[Any, int, dict[str, Any]]:
    """(scorer, number of features, summary) of a saved LambdaMART or neural model."""

    if path.suffix == ".pt":
        from learning_to_rank.neural import NeuralRanker

        nn = NeuralRanker.load(path)
        assert nn.scaler is not None
        info: dict[str, Any] = {
            "type": "neural",
            "loss": nn.params.loss,
            "hidden": list(nn.params.hidden),
        }
        return nn, int(nn.scaler.mean.shape[0]), info
    model = LambdaMART.load(path)
    assert model.ensemble is not None
    gain = model.ensemble.feature_importance("gain")
    top = np.argsort(-gain, kind="stable")[:10]
    summary: dict[str, Any] = {
        "type": "lambdamart",
        "n_trees": model.ensemble.n_trees,
        "objective": model.params.objective,
        "top_features_by_gain": [{"feature": int(j) + 1, "gain": float(gain[j])} for j in top],
    }
    return model, model.ensemble.n_features, summary


def _table(rows: list[tuple[str, dict[str, float]]], columns: Sequence[str]) -> str:
    head = f"{'':<24}" + "".join(f"{c:>10}" for c in columns)
    lines = [head]
    for name, m in rows:
        lines.append(f"{name:<24}" + "".join(f"{m[c]:>10.4f}" for c in columns))
    return "\n".join(lines)


def cmd_info(_: argparse.Namespace) -> int:
    print(f"learning-to-rank {__version__}")
    print(f"rust core: {'yes' if rust_available() else 'no'}")
    if rust_available():
        print(f"threads: {extension().num_threads()}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from learning_to_rank.datasets import prepare_mslr

    stats = prepare_mslr(args.source, args.out)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.train_file is not None:
        train = read_svmlight(args.train_file)
        valid = read_svmlight(args.valid_file) if args.valid_file else None
        test = None
    else:
        train, valid, test = _mslr(args)
    print(
        f"train: {train.n_queries:,} queries, {train.n_docs:,} documents, "
        f"{train.n_features} features"
    )
    t0 = time.perf_counter()
    if args.ranker == "lambdamart":
        params = GBDTParams(
            objective=args.objective,
            n_estimators=args.trees,
            learning_rate=args.learning_rate,
            num_leaves=args.leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            early_stopping_rounds=args.early_stopping,
            seed=args.seed,
        )
        model: Any = LambdaMART(params)

        def report(i: int, v: float | None) -> None:
            if i % args.log_every == 0 or i == params.n_estimators:
                val = f"  valid ndcg@10 {v:.4f}" if v is not None else ""
                print(f"tree {i:>5}{val}  {time.perf_counter() - t0:7.1f}s", flush=True)

        model.fit(train, valid, callback=report)
        print(f"kept {model.result.best_iteration} trees, fit {model.result.fit_seconds:.1f}s")
    else:
        from learning_to_rank.neural import NeuralRanker

        model = NeuralRanker(loss=args.loss, epochs=args.epochs, seed=args.seed, device=args.device)

        def report_nn(epoch: int, loss: float, v: float | None) -> None:
            val = f"  valid ndcg@10 {v:.4f}" if v is not None else ""
            print(
                f"epoch {epoch:>3}  loss {loss:.4f}{val}  {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )

        model.fit(train, valid, callback=report_nn)
    model.save(args.model)
    print(f"saved {args.model}")
    if test is not None:
        print(_table([("test", evaluate(test, model.predict(test)))], METRIC_COLUMNS))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    data = _split(args)
    rows = []
    for path in args.model:
        model, _, _ = load_model(path)
        rows.append((path.stem, evaluate(data, model.predict(data.X))))
    print(f"{data.n_queries:,} queries, {data.n_docs:,} documents")
    print(_table(rows, METRIC_COLUMNS))
    if args.json:
        print(json.dumps(dict(rows), indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from learning_to_rank.significance import compare

    data = _split(args)
    (a, _, _), (b, _, _) = load_model(args.a), load_model(args.b)
    pa = per_query(data.y, a.predict(data.X), data.offsets)[args.metric]
    pb = per_query(data.y, b.predict(data.X), data.offsets)[args.metric]
    c = compare(pa, pb, metric=args.metric, n_permutations=args.permutations, seed=args.seed)
    print(f"{args.metric} on {c.n_queries:,} queries")
    print(f"  A {args.a.stem:<20} {c.mean_a:.4f}")
    print(f"  B {args.b.stem:<20} {c.mean_b:.4f}")
    print(f"  A - B {c.diff:+.4f}  95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}]")
    print(
        f"  paired t-test p = {c.t_p_value:.4g}   "
        f"randomization test p = {c.randomization_p_value:.4g}"
    )
    print(f"  per query: A wins {c.wins}, ties {c.ties}, loses {c.losses}")
    return 0


def cmd_clicks(args: argparse.Namespace) -> int:
    from learning_to_rank.clicks import (
        ClickConfig,
        fit_cascade,
        fit_ctr,
        fit_pbm,
        pbm_propensity,
        simulate_clicks,
        swap_propensity,
    )

    train, _, _ = _mslr(args)
    rng = np.random.default_rng(args.seed)
    seed_queries = rng.choice(
        train.n_queries, size=min(args.production_queries, train.n_queries), replace=False
    )
    production = LambdaMART(n_estimators=30, seed=args.seed).fit(train.subset(seed_queries))
    cfg = ClickConfig(model=args.click_model, eta=args.eta, swap_fraction=args.swap_fraction)
    t0 = time.perf_counter()
    log = simulate_clicks(train, production.predict(train), args.sessions, cfg, seed=args.seed)
    print(
        f"{log.n_sessions:,} sessions on {train.n_queries:,} queries "
        f"in {time.perf_counter() - t0:.1f}s ({args.click_model} users)"
    )
    print("CTR by rank     " + " ".join(f"{v:6.3f}" for v in log.clicks.mean(axis=0)))
    fit, held = log.split(0.8, seed=args.seed)
    models = [
        fit_pbm(fit, train.n_docs),
        fit_cascade(fit, train.n_docs),
        fit_ctr(fit, train.n_docs),
    ]
    if args.click_model == "pbm":
        print(
            "true theta      " + " ".join(f"{v:6.3f}" for v in pbm_propensity(cfg.top_k, cfg.eta))
        )
    if args.swap_fraction > 0:
        print("swap estimate   " + " ".join(f"{v:6.3f}" for v in swap_propensity(fit)))
    print("EM (PBM) theta  " + " ".join(f"{v:6.3f}" for v in models[0].relative_theta()))
    print("held-out sessions: log-likelihood and perplexity (lower is better)")
    for m in models:
        e = m.evaluate(held)
        print(f"  {m.name:<8} {e['log_likelihood']:8.4f} {e['perplexity']:8.4f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocking
    from learning_to_rank.service import App, serve

    model, n_features, info = load_model(args.model)
    serve(App(model, n_features, info), args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learning_to_rank", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="version and backend").set_defaults(func=cmd_info)

    p = sub.add_parser("prepare-mslr", help="convert the MSLR-WEB10K parquet mirror")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train a ranker")
    _add_data_args(p, split=False)
    p.add_argument("--train-file", type=Path, help="LETOR training file instead of --data")
    p.add_argument("--valid-file", type=Path)
    p.add_argument(
        "--model", type=Path, required=True, help="output (.json for LambdaMART, .pt for neural)"
    )
    p.add_argument("--ranker", choices=("lambdamart", "neural"), default="lambdamart")
    p.add_argument("--objective", choices=("lambdarank", "regression"), default="lambdarank")
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--min-data-in-leaf", type=int, default=20)
    p.add_argument("--feature-fraction", type=float, default=1.0)
    p.add_argument("--early-stopping", type=int, default=0)
    p.add_argument("--loss", default="listnet", help="neural loss")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="metrics of saved models")
    _add_data_args(p)
    p.add_argument("--model", type=Path, nargs="+", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="paired significance test between two models")
    _add_data_args(p)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--metric", default="ndcg@10")
    p.add_argument("--permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("clicks", help="simulate clicks, fit click models, estimate position bias")
    _add_data_args(p, split=False)
    p.add_argument("--sessions", type=int, default=1_000_000)
    p.add_argument("--click-model", choices=("pbm", "cascade"), default="pbm")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--swap-fraction", type=float, default=0.1)
    p.add_argument("--production-queries", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_clicks)

    p = sub.add_parser("serve", help="HTTP ranking service")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
