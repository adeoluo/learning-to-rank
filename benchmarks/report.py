"""Print the Markdown tables used in the README and the evaluation report from the JSON results.

uv run python benchmarks/report.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

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
        + " | ".join(c.upper().replace("NDCG", "NDCG") for c in COLS)
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
    head = "| Model | " + " | ".join(COLS) + " | Train s |"
    return "\n".join([head, "|---|" + "---:|" * (len(COLS) + 1), *rows])


def main() -> None:
    print("## GBDT, MSLR-WEB10K, mean ± std over 5 folds\n")
    print(gbdt_table())
    print()
    print(gbdt_folds())
    print("\n## Neural\n")
    print(neural_table())
    for name in ("segments_mslr.json", "ultr_mslr.json", "throughput.json"):
        p = RESULTS / name
        if p.exists():
            print(f"\n## {name}\n")
            print(json.dumps(json.loads(p.read_text()), indent=1)[:4000])
    for p in sorted(Path(RESULTS).glob("istella_*.json")):
        d = json.loads(p.read_text())
        print(
            p.name,
            {c: round(d["metrics"][c], 4) for c in COLS},
            round(d["fit_seconds"]),
            round(d["peak_rss_mb"]),
        )


if __name__ == "__main__":
    main()
