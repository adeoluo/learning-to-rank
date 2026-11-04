"""Print the Markdown tables used in the README and the evaluation report from the JSON results.

uv run python benchmarks/report.py
"""

from __future__ import annotations

import json
import statistics as st
from typing import Any

import numpy as np
from common import RESULTS

COLS = ("ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "map", "mrr", "err@10")


def load(pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def ms(values: list[float], digits: int = 4) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{st.mean(values):.{digits}f} ± {st.stdev(values):.{digits}f}"


def gbdt_table() -> str:
    rows = []
    names = {
        "ours": "LambdaMART, this package (Rust core)",
        "lightgbm": "LightGBM lambdarank",
        "xgboost": "XGBoost rank:ndcg",
    }
    for lib, label in names.items():
        runs = load(f"gbdt_mslr_{lib}_fold*.json")
        if len(runs) != 5:
            continue
        cells = [ms([r["metrics"][c] for r in runs]) for c in COLS]
        fit = ms([r["fit_seconds"] for r in runs], 1)
        rss = f"{max(r['peak_rss_mb'] for r in runs):,.0f}"
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {fit} | {rss} |")
    head = (
        "| Model | "
        + " | ".join(c.upper() for c in COLS)
        + " | Train s (500 trees) | Peak RSS MB |"
    )
    sep = "|---|" + "---:|" * (len(COLS) + 2)
    return "\n".join([head, sep, *rows])


def gbdt_folds() -> str:
    lines = [
        "| Fold | Train docs | Test queries | This package | LightGBM | XGBoost "
        "| Fit s: ours / LightGBM / XGBoost |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for f in range(1, 6):
        r = {
            lib: json.loads((RESULTS / f"gbdt_mslr_{lib}_fold{f}.json").read_text())
            for lib in ("ours", "lightgbm", "xgboost")
            if (RESULTS / f"gbdt_mslr_{lib}_fold{f}.json").exists()
        }
        if len(r) < 3:
            continue
        lines.append(
            f"| {f} | {r['ours']['train_docs']:,} | {r['ours']['test_queries']:,} | "
            + " | ".join(
                f"{r[k]['metrics']['ndcg@10']:.4f}" for k in ("ours", "lightgbm", "xgboost")
            )
            + " | "
            + " / ".join(f"{r[k]['fit_seconds']:.0f}" for k in ("ours", "lightgbm", "xgboost"))
            + " |"
        )
    return "\n".join(lines)


def neural_table() -> str:
    rows = []
    for loss in ("pointwise", "ranknet", "lambdarank", "listnet"):
        runs = load(f"neural_mslr_{loss}_fold*.json")
        if len(runs) != 5:
            continue
        cells = [ms([r["metrics"][c] for r in runs]) for c in COLS]
        rows.append(
            f"| MLP, {loss} | "
            + " | ".join(cells)
            + f" | {ms([r['fit_seconds'] for r in runs], 0)} |"
        )
    head = "| Model | " + " | ".join(c.upper() for c in COLS) + " | Train s |"
    return "\n".join([head, "|---|" + "---:|" * (len(COLS) + 1), *rows])


def significance_table() -> str:
    d = json.loads((RESULTS / "segments_mslr.json").read_text())
    lines = [
        f"| System vs {d['baseline']} | Mean NDCG@10 | Difference | 95% CI | t-test p "
        "| Randomization p | Wins / ties / losses |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for name, r in d["overall_vs_baseline"].items():
        lines.append(
            f"| {name} | {r['mean_a']:.4f} | {r['diff']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['t_p_value']:.3g} | "
            f"{r['randomization_p_value']:.3g} | {r['wins']:,} / {r['ties']:,} / {r['losses']:,} |"
        )
    return "\n".join(lines)


def segments_table(key: str) -> str:
    d = json.loads((RESULTS / "segments_mslr.json").read_text())
    systems = d["systems"]
    head = "| Segment | Queries | " + " | ".join(systems) + " |"
    lines = [head, "|---|---:|" + "---:|" * len(systems)]
    for row in d[key]:
        cells = []
        for name in systems:
            cell = f"{row['means'][name]:.4f}"
            p = row["adjusted_p_values"].get(name)
            if p is not None and p < 0.05:
                cell += " *"
            cells.append(cell)
        lines.append(f"| {row['segment']} | {row['n_queries']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ULTR_ROWS = (
    ("production", "Production ranker (30 trees, 60 queries)"),
    ("naive", "Naive: click-through rate as target"),
    ("ipw_true", "IPW, true propensities"),
    ("ipw_swap", "IPW, swap-intervention propensities"),
    ("ipw_em", "IPW, EM (PBM) propensities"),
    ("skyline", "Skyline: true labels of the shown documents"),
)


def ultr_table() -> str:
    d = json.loads((RESULTS / "ultr_mslr.json").read_text())
    etas = d["protocol"]["etas"]
    head = "| Training signal | " + " | ".join(
        f"NDCG@10, eta {e:g} | ERR@10, eta {e:g}" for e in etas
    )
    lines = [head + " |", "|---|" + "---:|" * (2 * len(etas))]
    for key, label in ULTR_ROWS:
        cells = []
        for e in etas:
            rs = [r for r in d["runs"] if r["eta"] == e]
            cells += [ms([r[key]["ndcg@10"] for r in rs]), ms([r[key]["err@10"] for r in rs])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    full = d["supervised_full_labels"]
    lines.append(
        "| Supervised, all 723,412 labelled training documents (500 trees) | "
        + " | ".join(f"{full['ndcg@10']:.4f} | {full['err@10']:.4f}" for _ in etas)
        + " |"
    )
    return "\n".join(lines)


def click_model_table() -> str:
    d = json.loads((RESULTS / "ultr_mslr.json").read_text())["click_models"]
    lines = [
        "| Users simulated as | PBM (EM) | Cascade | CTR by rank |",
        "|---|---:|---:|---:|",
    ]
    for user, models in d.items():
        cells = [f"{models[m]['perplexity']:.4f}" for m in ("pbm", "cascade", "ctr")]
        lines.append(f"| {user} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def throughput_table() -> str:
    d = json.loads((RESULTS / "throughput.json").read_text())
    by = {(r["library"], r["threads"]): r["seconds"] for r in d["scaling"]}
    threads = sorted({t for _, t in by})
    lines = [
        "| Threads | " + " | ".join(str(t) for t in threads) + " |",
        "|---|" + "---:|" * len(threads),
    ]
    for lib, label in (("ours", "This package, s"), ("lightgbm", "LightGBM, s")):
        lines.append(f"| {label} | " + " | ".join(f"{by[(lib, t)]:.1f}" for t in threads) + " |")
    return "\n".join(lines)


def istella_table() -> str:
    names = {
        "ours": "LambdaMART, this package (Rust core)",
        "lightgbm": "LightGBM lambdarank",
        "xgboost": "XGBoost rank:ndcg",
    }
    cols = ("ndcg@1", "ndcg@5", "ndcg@10", "map", "mrr", "err@10")
    lines = [
        "| Model | "
        + " | ".join(c.upper() for c in cols)
        + " | Train s (500 trees) | Peak RSS MB |",
        "|---|" + "---:|" * (len(cols) + 2),
    ]
    for lib, label in names.items():
        p = RESULTS / f"istella_{lib}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cells = [f"{d['metrics'][c]:.4f}" for c in cols]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {d['fit_seconds']:.0f} | {d['peak_rss_mb']:,.0f} |"
        )
    return "\n".join(lines)


def istella_tests() -> str:
    from learning_to_rank.significance import compare

    runs = {
        lib: json.loads((RESULTS / f"istella_{lib}.json").read_text())
        for lib in ("ours", "lightgbm", "xgboost")
    }
    base = np.asarray(runs["lightgbm"]["per_query_ndcg@10"])
    lines = [
        "| System vs LightGBM | Difference | 95% CI | t-test p | Randomization p |",
        "|---|---:|---|---:|---:|",
    ]
    for lib, label in (("ours", "LambdaMART (this package)"), ("xgboost", "XGBoost")):
        r = compare(np.asarray(runs[lib]["per_query_ndcg@10"]), base).to_dict()
        lines.append(
            f"| {label} | {r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{r['t_p_value']:.3g} | {r['randomization_p_value']:.3g} |"
        )
    return "\n".join(lines)


def main() -> None:
    sections = [
        ("GBDT, MSLR-WEB10K, mean ± std over 5 folds", gbdt_table),
        ("GBDT, MSLR-WEB10K, per fold", gbdt_folds),
        ("Neural rankers, MSLR-WEB10K, mean ± std over 5 folds", neural_table),
        ("Paired tests over all 10,000 queries", significance_table),
        ("Segments by popularity", lambda: segments_table("popularity")),
        ("Segments by query length", lambda: segments_table("query_length")),
        ("Unbiased learning to rank, mean ± std over 3 seeds", ultr_table),
        ("Click models, held-out perplexity", click_model_table),
        ("Thread scaling, 50 trees", throughput_table),
        ("Istella-S", istella_table),
        ("Istella-S, paired tests over the test queries", istella_tests),
    ]
    for title, fn in sections:
        try:
            table = fn()
        except FileNotFoundError:
            continue
        print(f"## {title}\n\n{table}\n")


if __name__ == "__main__":
    main()
