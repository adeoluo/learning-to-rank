//! Feature binning: per-feature bin upper edges fitted on a row sample, and the mapping from raw
//! values to `u8` bin indices that the histogram tree learner works on.
//!
//! Edges are actual sample values: bin `b` holds values in `(edges[b-1], edges[b]]` and the last
//! bin holds everything above the last edge, so `bin(x) <= t` exactly when `x <= edges[t]`. NaN
//! maps to bin 0. The numpy reference (`learning_to_rank.binning`) uses the same greedy rule.

use rayon::prelude::*;

/// Fit the edges of one feature from its sample values (any order).
///
/// With at most `max_bin` distinct values every distinct value gets its own bin. Otherwise a greedy
/// pass over the sorted distinct values closes a bin once it holds at least
/// `remaining_count / remaining_bins` samples, so heavy values get their own bins and identical
/// values never straddle a boundary.
pub fn fit_edges(values: &mut [f32], max_bin: usize) -> Vec<f32> {
    assert!(max_bin >= 2, "max_bin must be at least 2");
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).expect("NaN filtered"));
    let mut distinct: Vec<f32> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &v in &finite {
        if distinct.last() == Some(&v) {
            *counts.last_mut().expect("non-empty") += 1;
        } else {
            distinct.push(v);
            counts.push(1);
        }
    }
    if distinct.len() <= 1 {
        return Vec::new();
    }
    if distinct.len() <= max_bin {
        return distinct[..distinct.len() - 1].to_vec();
    }
    let mut edges = Vec::with_capacity(max_bin - 1);
    let mut remaining_count = finite.len();
    let mut remaining_bins = max_bin;
    let mut in_bin = 0usize;
    let last = distinct.len() - 1;
    for (i, (&v, &c)) in distinct.iter().zip(counts.iter()).enumerate() {
        if i == last || remaining_bins == 1 {
            break;
        }
        in_bin += c;
        // Integer form of in_bin >= remaining_count / remaining_bins (no float rounding).
        if in_bin * remaining_bins >= remaining_count {
            edges.push(v);
            remaining_count -= in_bin;
            remaining_bins -= 1;
            in_bin = 0;
        }
    }
    edges
}

/// Fit edges for every column of a row-major `n x d` matrix from the rows in `sample`.
pub fn fit_all(x: &[f32], d: usize, sample: &[u32], max_bin: usize) -> Vec<Vec<f32>> {
    (0..d)
        .into_par_iter()
        .map(|j| {
            let mut col: Vec<f32> = sample.iter().map(|&r| x[r as usize * d + j]).collect();
            fit_edges(&mut col, max_bin)
        })
        .collect()
}

/// Bin index of `value` given sorted edges: the number of edges strictly below it.
#[inline]
pub fn bin_of(edges: &[f32], value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    edges.partition_point(|&e| e < value) as u8
}

/// Map a row-major `n x d` float matrix to a row-major `n x d` bin matrix, in parallel.
pub fn apply(x: &[f32], d: usize, edges: &[Vec<f32>], out: &mut [u8]) {
    assert_eq!(edges.len(), d, "one edge list per feature");
    assert_eq!(x.len(), out.len(), "output must match the input size");
    if d == 0 {
        return;
    }
    out.par_chunks_mut(d * 1024)
        .zip(x.par_chunks(d * 1024))
        .for_each(|(o, xs)| {
            for (row_out, row_in) in o.chunks_mut(d).zip(xs.chunks(d)) {
                for j in 0..d {
                    row_out[j] = bin_of(&edges[j], row_in[j]);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_distinct_values_get_one_bin_each() {
        let mut v = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        assert_eq!(fit_edges(&mut v, 255), vec![1.0, 2.0]);
        let e = vec![1.0, 2.0];
        assert_eq!(bin_of(&e, 0.5), 0);
        assert_eq!(bin_of(&e, 1.0), 0);
        assert_eq!(bin_of(&e, 1.5), 1);
        assert_eq!(bin_of(&e, 2.0), 1);
        assert_eq!(bin_of(&e, 9.0), 2);
        assert_eq!(bin_of(&e, f32::NAN), 0);
    }

    #[test]
    fn constant_and_empty_features_have_no_edges() {
        assert!(fit_edges(&mut [4.0, 4.0], 16).is_empty());
        assert!(fit_edges(&mut [], 16).is_empty());
        assert!(fit_edges(&mut [f32::NAN, 1.0], 16).is_empty());
    }

    #[test]
    fn many_values_respect_max_bin_and_order() {
        let mut v: Vec<f32> = (0..10_000).map(|i| ((i * 7919) % 10_007) as f32).collect();
        let e = fit_edges(&mut v, 64);
        assert!(e.len() <= 63 && e.len() >= 60, "{}", e.len());
        assert!(e.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn heavy_value_gets_its_own_bin() {
        let mut v = vec![0.0f32; 900];
        v.extend((1..=100).map(|i| i as f32));
        let e = fit_edges(&mut v, 8);
        assert_eq!(e[0], 0.0);
        assert!(e.len() <= 7);
    }

    #[test]
    fn apply_matches_bin_of() {
        let x = vec![0.0, 5.0, 1.0, 6.0, 2.0, 7.0];
        let edges = vec![vec![0.0, 1.0], vec![6.0]];
        let mut out = vec![0u8; 6];
        apply(&x, 2, &edges, &mut out);
        assert_eq!(out, vec![0, 0, 1, 0, 2, 1]);
        let s: Vec<u32> = (0..3).collect();
        let fitted = fit_all(&x, 2, &s, 255);
        assert_eq!(fitted, vec![vec![0.0, 1.0], vec![5.0, 6.0]]);
    }
}
