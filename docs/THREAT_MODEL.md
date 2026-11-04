# Threat Model - Learning To Rank

This is an offline training and evaluation library with a small scoring service. It reads
datasets and model files chosen by the operator and feature vectors sent to the service; it
calls no external services at run time.

## Assets

- **Ranking integrity:** the order a deployed model returns; manipulated scores or models change
  what users see first.
- **Evaluation integrity:** reported metrics and significance results drive model choices.
- **Service availability** of `/rank` and `/score`.

## Trust boundaries

1. Feature vectors arriving at the HTTP service are untrusted.
2. Model files loaded by `serve`, `evaluate` and `compare` are trusted to come from the operator
   but are parsed defensively.
3. Datasets come from the Hugging Face mirror and the Istella server via `scripts/download_data.sh`.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Resource exhaustion | Large bodies or many documents per request | Memory or CPU exhaustion | Body size capped (64 MB), at most 100,000 documents per request, threaded server |
| Malformed input | Wrong feature count, strings, NaN or infinity | Crashes or garbage scores | Shape and finiteness validation with 400 responses; covered by tests |
| Code execution through model files | Pickled objects in a checkpoint | Arbitrary code on load | Trees are JSON; neural checkpoints load with `torch.load(weights_only=True)` |
| Malformed tree files | Out-of-range features, cyclic child pointers | Out-of-bounds reads or infinite loops | `Ensemble.from_dict` and the Rust loader check offsets, feature ranges and that children point forward |
| Feature manipulation (search spam) | A document crafted to score high on known features | Low-quality results ranked first | Out of scope for the library; segment analysis and per-query comparisons help spot shifts |
| Click spam or bots | Fake clicks in a log used for IPW training | Model learns the attack | Simulation only here; real logs need bot filtering and click caps (`max_weight`) |
| Biased offline conclusions | Metric convention mismatch, cherry-picked folds | Wrong model choice | One metric module scores every library; all five folds reported; paired tests with Holm adjustment |
| Supply chain | Malicious dependency or dataset | Code execution or corrupted results | Dependencies pinned in `uv.lock`; CodeQL in CI; datasets are numeric arrays parsed by our own readers |

## Residual risk

The service has no authentication or rate limiting; run it behind a gateway that provides them.
Adversarial ranking (spam) and click fraud are acknowledged and not defended against.
