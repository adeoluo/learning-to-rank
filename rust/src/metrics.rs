//! Ranking metrics used inside training (validation NDCG for early stopping).
//!
//! The full metric suite (NDCG, MAP, MRR, ERR) lives in the vectorized numpy module
//! `learning_to_rank.metrics`; this one mirrors its NDCG exactly: gain `2^label - 1`, discount
//! `1 / log2(rank + 1)`, ties broken by document order, and NDCG = 1 for a query whose ideal DCG
//! is 0 (the LightGBM and XGBoost convention).

use rayon::prelude::*;

#[inline]
pub fn gain(label: f32) -> f64 {
    (label as f64).exp2() - 1.0
}

#[inline]
pub fn discount(rank0: usize) -> f64 {
    1.0 / ((rank0 as f64) + 2.0).log2()
}

/// Indices `0..n` sorted by descending score; equal scores keep document order.
pub fn argsort_desc(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Ideal DCG@k of one query's labels.
pub fn ideal_dcg(labels: &[f32], k: usize) -> f64 {
    let mut sorted: Vec<f32> = labels.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted
        .iter()
        .take(k)
        .enumerate()
        .map(|(r, &l)| gain(l) * discount(r))
        .sum()
}

/// NDCG@k of one query.
pub fn ndcg_query(labels: &[f32], scores: &[f64], k: usize) -> f64 {
    let idcg = ideal_dcg(labels, k);
    if idcg <= 0.0 {
        return 1.0;
    }
    let order = argsort_desc(scores);
    let dcg: f64 = order
        .iter()
        .take(k)
        .enumerate()
        .map(|(r, &i)| gain(labels[i]) * discount(r))
        .sum();
    dcg / idcg
}

/// Mean NDCG@k over the queries delimited by `offsets` (length `n_queries + 1`).
pub fn mean_ndcg(labels: &[f32], scores: &[f64], offsets: &[usize], k: usize) -> f64 {
    let nq = offsets.len().saturating_sub(1);
    if nq == 0 {
        return 0.0;
    }
    let total: f64 = (0..nq)
        .into_par_iter()
        .map(|q| {
            let (s, e) = (offsets[q], offsets[q + 1]);
            ndcg_query(&labels[s..e], &scores[s..e], k)
        })
        .collect::<Vec<f64>>()
        .iter()
        .sum();
    total / nq as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_reversed_rankings() {
        let labels = [3.0, 2.0, 0.0];
        assert!((ndcg_query(&labels, &[3.0, 2.0, 1.0], 10) - 1.0).abs() < 1e-12);
        let rev = ndcg_query(&labels, &[1.0, 2.0, 3.0], 10);
        let dcg = 0.0 + 3.0 / 3f64.log2() + 7.0 / 4f64.log2();
        let idcg = 7.0 + 3.0 / 3f64.log2();
        assert!((rev - dcg / idcg).abs() < 1e-12);
    }

    #[test]
    fn empty_query_scores_one_and_ties_keep_order() {
        assert_eq!(ndcg_query(&[0.0, 0.0], &[1.0, 2.0], 5), 1.0);
        // Tied scores: the first document stays first.
        let v = ndcg_query(&[0.0, 1.0], &[1.0, 1.0], 1);
        assert_eq!(v, 0.0);
        let offsets = [0, 2, 4];
        let m = mean_ndcg(&[0.0, 1.0, 1.0, 0.0], &[0.0, 1.0, 1.0, 0.0], &offsets, 1);
        assert!((m - 1.0).abs() < 1e-12);
        assert_eq!(mean_ndcg(&[], &[], &[0], 3), 0.0);
    }
}
