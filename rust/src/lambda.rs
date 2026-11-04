//! LambdaRank gradients and hessians (Burges 2010), with the normalization choices LightGBM makes:
//! pairs are formed between the top `truncation_level` positions of the current ranking and every
//! position below them, each pair's lambda is scaled by `|delta NDCG|`, optionally divided by
//! `0.01 + |score gap|`, and the query's lambdas are rescaled by `log2(1 + S) / S` where `S` is
//! their total magnitude. Queries run in parallel; each writes only its own slice.

use rayon::prelude::*;

use crate::metrics::{argsort_desc, discount, gain, ideal_dcg};

/// Convenience for tests and one-off calls: gains and discounts computed here.
pub fn gradients_once(
    labels: &[f32],
    scores: &[f64],
    offsets: &[usize],
    p: &LambdaParams,
    g: &mut [f64],
    h: &mut [f64],
) {
    let inv = inverse_max_dcg(labels, offsets, p.truncation_level);
    let gains = gain_table(labels);
    let longest = offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    let discounts = discount_table(longest);
    all_gradients(labels, &gains, scores, offsets, &inv, &discounts, p, g, h);
}

#[derive(Debug, Clone, Copy)]
pub struct LambdaParams {
    pub truncation_level: usize,
    pub sigma: f64,
    pub norm: bool,
}

impl Default for LambdaParams {
    fn default() -> Self {
        Self {
            truncation_level: 30,
            sigma: 1.0,
            norm: true,
        }
    }
}

/// `1 / log2(rank + 2)` for ranks `0..n` (shared by all queries).
pub fn discount_table(n: usize) -> Vec<f64> {
    (0..n).map(discount).collect()
}

/// `2^label - 1` per document.
pub fn gain_table(labels: &[f32]) -> Vec<f64> {
    labels.iter().map(|&l| gain(l)).collect()
}

/// Gradients for one query; `g` and `h` are overwritten. `gains` holds `2^label - 1` per
/// document and `discounts` at least `labels.len()` rank discounts.
#[allow(clippy::too_many_arguments)]
pub fn query_gradients(
    labels: &[f32],
    gains: &[f64],
    scores: &[f64],
    inv_max_dcg: f64,
    discounts: &[f64],
    p: &LambdaParams,
    g: &mut [f64],
    h: &mut [f64],
) {
    g.iter_mut().for_each(|v| *v = 0.0);
    h.iter_mut().for_each(|v| *v = 0.0);
    let cnt = labels.len();
    if cnt < 2 || inv_max_dcg <= 0.0 {
        return;
    }
    let sorted = argsort_desc(scores);
    let best = scores[sorted[0]];
    let worst = scores[sorted[cnt - 1]];
    let mut sum_lambdas = 0.0f64;
    let top = p.truncation_level.min(cnt - 1);
    for i in 0..top {
        for j in (i + 1)..cnt {
            let (li, lj) = (labels[sorted[i]], labels[sorted[j]]);
            if li == lj {
                continue;
            }
            let (high_rank, low_rank) = if li > lj { (i, j) } else { (j, i) };
            let high = sorted[high_rank];
            let low = sorted[low_rank];
            let delta_score = scores[high] - scores[low];
            let dcg_gap = gains[high] - gains[low];
            let paired_discount = (discounts[high_rank] - discounts[low_rank]).abs();
            let mut delta_ndcg = dcg_gap * paired_discount * inv_max_dcg;
            if p.norm && best != worst {
                delta_ndcg /= 0.01 + delta_score.abs();
            }
            let mut p_lambda = 1.0 / (1.0 + (p.sigma * delta_score).exp());
            let mut p_hessian = p_lambda * (1.0 - p_lambda);
            p_lambda *= -p.sigma * delta_ndcg;
            p_hessian *= p.sigma * p.sigma * delta_ndcg;
            g[low] -= p_lambda;
            h[low] += p_hessian;
            g[high] += p_lambda;
            h[high] += p_hessian;
            sum_lambdas -= 2.0 * p_lambda;
        }
    }
    if p.norm && sum_lambdas > 0.0 {
        let factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
        g.iter_mut().for_each(|v| *v *= factor);
        h.iter_mut().for_each(|v| *v *= factor);
    }
}

/// `1 / maxDCG@truncation_level` per query (0 for queries without relevant documents).
pub fn inverse_max_dcg(labels: &[f32], offsets: &[usize], truncation_level: usize) -> Vec<f64> {
    (0..offsets.len().saturating_sub(1))
        .into_par_iter()
        .map(|q| {
            let idcg = ideal_dcg(&labels[offsets[q]..offsets[q + 1]], truncation_level);
            if idcg > 0.0 {
                1.0 / idcg
            } else {
                0.0
            }
        })
        .collect()
}

/// Split `v` into the per-query mutable slices delimited by `offsets`.
pub fn split_by_offsets<'a, T>(mut v: &'a mut [T], offsets: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(offsets.len().saturating_sub(1));
    for w in offsets.windows(2) {
        let (head, tail) = v.split_at_mut(w[1] - w[0]);
        out.push(head);
        v = tail;
    }
    out
}

/// Gradients for all queries, in parallel.
#[allow(clippy::too_many_arguments)]
pub fn all_gradients(
    labels: &[f32],
    gains: &[f64],
    scores: &[f64],
    offsets: &[usize],
    inv_max_dcg: &[f64],
    discounts: &[f64],
    p: &LambdaParams,
    g: &mut [f64],
    h: &mut [f64],
) {
    let gs = split_by_offsets(g, offsets);
    let hs = split_by_offsets(h, offsets);
    gs.into_par_iter()
        .zip(hs.into_par_iter())
        .enumerate()
        .for_each(|(q, (gq, hq))| {
            let (s, e) = (offsets[q], offsets[q + 1]);
            query_gradients(
                &labels[s..e],
                &gains[s..e],
                &scores[s..e],
                inv_max_dcg[q],
                discounts,
                p,
                gq,
                hq,
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(labels: &[f32], scores: &[f64], inv: f64, p: &LambdaParams, g: &mut [f64], h: &mut [f64]) {
        let gains = gain_table(labels);
        let discounts = discount_table(labels.len());
        query_gradients(labels, &gains, scores, inv, &discounts, p, g, h);
    }

    #[test]
    fn relevant_document_is_pushed_up() {
        let labels = [0.0, 2.0, 1.0];
        let scores = [0.0, 0.0, 0.0];
        let inv = 1.0 / ideal_dcg(&labels, 30);
        let (mut g, mut h) = (vec![0.0; 3], vec![0.0; 3]);
        q(
            &labels,
            &scores,
            inv,
            &LambdaParams::default(),
            &mut g,
            &mut h,
        );
        assert!(g[1] < 0.0, "negative gradient raises the score: {g:?}");
        assert!(g[0] > 0.0);
        assert!(h.iter().all(|&v| v > 0.0));
        assert!(g.iter().sum::<f64>().abs() < 1e-12, "lambdas cancel");
    }

    #[test]
    fn empty_or_uniform_queries_have_zero_gradient() {
        let (mut g, mut h) = (vec![1.0; 2], vec![1.0; 2]);
        q(
            &[1.0, 1.0],
            &[0.3, 0.1],
            0.5,
            &LambdaParams::default(),
            &mut g,
            &mut h,
        );
        assert_eq!(g, vec![0.0, 0.0]);
        q(
            &[0.0, 1.0],
            &[0.3, 0.1],
            0.0,
            &LambdaParams::default(),
            &mut g,
            &mut h,
        );
        assert_eq!(h, vec![0.0, 0.0]);
    }

    #[test]
    fn unnormalized_matches_closed_form() {
        // One pair: lambda = -sigma * |dNDCG| / (1 + exp(sigma * (s_high - s_low))).
        let labels = [1.0, 0.0];
        let scores = [0.2, 0.5];
        let p = LambdaParams {
            truncation_level: 30,
            sigma: 2.0,
            norm: false,
        };
        let inv = 1.0 / ideal_dcg(&labels, 30);
        let (mut g, mut h) = (vec![0.0; 2], vec![0.0; 2]);
        q(&labels, &scores, inv, &p, &mut g, &mut h);
        let dndcg = 1.0 * (1.0 - 1.0 / 3f64.log2()) * inv;
        let rho = 1.0 / (1.0 + (2.0f64 * (0.2 - 0.5)).exp());
        assert!((g[0] - (-2.0 * dndcg * rho)).abs() < 1e-12);
        assert!((h[0] - 4.0 * dndcg * rho * (1.0 - rho)).abs() < 1e-12);
    }

    #[test]
    fn all_gradients_handles_many_queries() {
        let labels = [1.0, 0.0, 0.0, 2.0, 0.0];
        let scores = [0.0; 5];
        let offsets = [0, 2, 5];
        let (mut g, mut h) = (vec![0.0; 5], vec![0.0; 5]);
        gradients_once(
            &labels,
            &scores,
            &offsets,
            &LambdaParams::default(),
            &mut g,
            &mut h,
        );
        assert!(g[0] < 0.0 && g[3] < 0.0);
    }
}
