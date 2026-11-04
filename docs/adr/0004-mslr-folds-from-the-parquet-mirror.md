# ADR-0004: Rebuild the five MSLR-WEB10K folds from the Fold 1 parquet mirror

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

MSLR-WEB10K is published by Microsoft as one archive with five folds built from partitions
S1..S5 (fold k trains on three partitions, validates on the next and tests on the last). The
official OneDrive link now redirects to a SharePoint page that refuses anonymous downloads. The
only public mirror we found is the Hugging Face dataset `philipphager/MSLR-WEB10k`, which holds
Fold 1 as three parquet files (train, validation, test).

## Decision

Reconstruct the partitions from query ids. Every MSLR-WEB10K query id is congruent to 1, 4, 7,
10 or 13 modulo 15. In the Fold 1 files the validation split (S4) holds only ids = 10 (mod 15),
the test split (S5) only 13, and the training split holds 1, 4 and 7 with exactly 2,000 queries
each. We therefore map residues 1, 4, 7, 10, 13 to S1..S5, store all 1,200,192 rows once with
their partition (`prepare_mslr`), and assemble fold k as train = S(k), S(k+1), S(k+2),
validation = S(k+3), test = S(k+4), as in the official layout.

## Alternatives considered

- **Fold 1 only.** Simpler, but one fold gives no spread across folds and only 2,000 test
  queries for segment analysis.
- **Random re-splits.** Not comparable with published MSLR results.

## Consequences

- The Fold 1 row counts reproduce the published 723,412 / 235,259 / 241,521. The other folds'
  assignment rests on the residue pattern; it is exact if the published partitions are the
  residue classes, which the Fold 1 files show for S4, S5 and the union of S1 to S3.
- Every query is a test query in exactly one fold, so the five test sets together cover all
  10,000 queries once (used by the segment analysis).
- `tests/test_datasets.py` checks the residue mapping and the fold assembly on a synthetic
  parquet mirror.
