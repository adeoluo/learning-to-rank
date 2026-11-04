# Data Card - Learning To Rank

## Sources and provenance

| Dataset | Source | Size used | Retrieved |
|---|---|---|---|
| MSLR-WEB10K | Qin and Liu, *Introducing LETOR 4.0 Datasets* (2013), Microsoft Research. Fold 1 parquet mirror on Hugging Face: `philipphager/MSLR-WEB10k` (`train`, `validation`, `test`, 431 MB) | 10,000 queries, 1,200,192 judged query-document pairs, 136 features, all 5 folds | 2026-11-04 |
| Istella-S LETOR | Lucchese et al., *Post-Learning Optimization of Tree Ensembles for Efficient Ranking* (SIGIR 2016), Istella S.r.l., `http://library.istella.it/dataset/istella-s-letor.tar.gz` (472 MB) | 33,018 queries, 3,408,630 pairs, 220 features, train / validation / test | 2026-11-04 |
| Synthetic | `learning_to_rank.data.synthetic` (seeded) | used in tests and CI only | generated |

`scripts/download_data.sh` fetches both public datasets into `../.bench-data/learning-to-rank`;
nothing from them is committed. Click logs are simulated from the MSLR labels
(`learning_to_rank.clicks`); no real user data is used.

## Licensing

- MSLR-WEB10K is released by Microsoft for research use; the mirror republishes the Fold 1 files.
- Istella-S is distributed under the Istella LETOR licence agreement: non-commercial use only
  (research, teaching, personal experimentation). This repository uses it for non-commercial
  benchmarking, does not redistribute it, and cites the authors as the licence asks.

## Composition

- **MSLR-WEB10K:** Bing queries (ids only, no text) with up to 1,251 judged URLs each (mean 120),
  relevance grades 0 to 4 (52 percent are 0). The 136 features are numeric: term statistics for
  body, anchor, title, URL and whole document (TF, IDF, BM25, language-model scores), link and
  quality scores, and three click and dwell features from Bing logs. Partitions S1..S5 have 2,000
  queries each.
- **Istella-S:** Istella queries with 220 anonymized numeric features and grades 0 to 4; on
  average 103 documents per query after Istella sampled irrelevant pairs.

## Preprocessing

- MSLR: the parquet files are converted to float32 `X.npy` (1,200,192 x 136), `y.npy`, `qid.npy`
  and `part.npy`, sorted by partition and query id. The partition of each query is its id modulo
  15 (residues 1, 4, 7, 10, 13 are S1..S5), reconstructed and justified in
  [ADR-0004](adr/0004-mslr-folds-from-the-parquet-mirror.md). Fold 1 row counts match the
  published 723,412 / 235,259 / 241,521.
- Istella-S: streamed from the archive and stored as 255-bin `uint8` matrices with edges fitted
  on every tenth training row ([ADR-0005](adr/0005-istella-as-shared-binned-matrices.md)).
- Neural rankers apply `sign(x) log1p(|x|)` and standardization fitted on the training split.

## Contamination checks

Folds use disjoint query sets: a query is in exactly one of train, validation and test of each
fold. Bin edges, feature scalers and early stopping use training and validation data only. The
segment analysis concatenates the five test folds, which cover each query once.

## Known biases and gaps

- Judgments are graded by annotators on a pool of candidate URLs; documents outside the pool are
  not judged, and grade 0 dominates.
- MSLR has no query text or query frequency. The head / torso / tail split uses the summed
  query-url click counts of each query's documents (feature 134) as a popularity signal, and
  query length is recovered from features 5 and 10; both are proxies.
- Both datasets are web search in 2010 to 2016 (English for MSLR, Italian for Istella); results
  may not transfer to product search or ads.

## Ethical considerations

Queries and documents are anonymized feature vectors; no personal data is present. Simulated
clicks are generated from relevance labels with documented user models.
