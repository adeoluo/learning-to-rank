//! Gradient boosting driver: LambdaMART (LambdaRank gradients) or squared-error regression,
//! histogram trees, feature subsampling, validation NDCG and early stopping.

use rayon::prelude::*;

use crate::lambda::{LambdaParams, all_gradients, discount_table, gain_table, inverse_max_dcg};
use crate::metrics::mean_ndcg;
use std::time::Instant;

use crate::tree::{HistLayout, Matrix, Timings, Tree, TreeParams, Workspace, grow_tree, transpose};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    LambdaRank,
    Regression,
}

#[derive(Debug, Clone, Copy)]
pub struct BoostParams {
    pub objective: Objective,
    pub learning_rate: f64,
    pub tree: TreeParams,
    pub lambda: LambdaParams,
    pub feature_fraction: f64,
    pub seed: u64,
    pub eval_at: usize,
}

impl Default for BoostParams {
    fn default() -> Self {
        Self {
            objective: Objective::LambdaRank,
            learning_rate: 0.1,
            tree: TreeParams::default(),
            lambda: LambdaParams::default(),
            feature_fraction: 1.0,
            seed: 0,
            eval_at: 10,
        }
    }
}

/// SplitMix64: small, fast and reproducible across platforms.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Choose `round(fraction * d)` (at least 1) features with a partial Fisher-Yates shuffle.
pub fn feature_mask(d: usize, fraction: f64, rng: &mut SplitMix64) -> Vec<bool> {
    if fraction >= 1.0 || d == 0 {
        return vec![true; d];
    }
    let k = ((fraction * d as f64).round() as usize).clamp(1, d);
    let mut perm: Vec<usize> = (0..d).collect();
    for i in 0..k {
        let j = i + rng.below((d - i) as u64) as usize;
        perm.swap(i, j);
    }
    let mut mask = vec![false; d];
    for &j in &perm[..k] {
        mask[j] = true;
    }
    mask
}

/// One dataset in binned form, owned so that a trainer can outlive the Python call.
#[derive(Debug, Clone)]
pub struct Binned {
    pub bins: Vec<u8>,
    pub n: usize,
    pub d: usize,
    pub labels: Vec<f32>,
    pub offsets: Vec<usize>,
}

impl Binned {
    pub fn validate(&self) -> Result<(), String> {
        if self.bins.len() != self.n * self.d {
            return Err(format!(
                "bins has {} values, expected {}",
                self.bins.len(),
                self.n * self.d
            ));
        }
        if self.labels.len() != self.n {
            return Err("labels must have one value per row".into());
        }
        if self.offsets.first() != Some(&0) || self.offsets.last() != Some(&self.n) {
            return Err("offsets must start at 0 and end at the row count".into());
        }
        if self.offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err("offsets must be non-decreasing".into());
        }
        Ok(())
    }
}

/// Stateful booster: one call to `step` adds one tree.
pub struct Trainer {
    pub params: BoostParams,
    pub train: Binned,
    pub valid: Option<Binned>,
    pub layout: HistLayout,
    pub edges: Vec<Vec<f32>>,
    pub trees: Vec<Tree>,
    pub train_scores: Vec<f64>,
    pub valid_scores: Vec<f64>,
    pub timings: Timings,
    columns: Vec<u8>,
    workspace: Workspace,
    inv_max_dcg: Vec<f64>,
    gains: Vec<f64>,
    discounts: Vec<f64>,
    grad: Vec<f64>,
    hess: Vec<f64>,
    indices: Vec<u32>,
    rng: SplitMix64,
}

impl Trainer {
    pub fn new(
        params: BoostParams,
        train: Binned,
        valid: Option<Binned>,
        edges: Vec<Vec<f32>>,
    ) -> Result<Self, String> {
        train.validate()?;
        if let Some(v) = &valid {
            v.validate()?;
            if v.d != train.d {
                return Err("validation data has a different number of features".into());
            }
        }
        if edges.len() != train.d {
            return Err("one edge list per feature is required".into());
        }
        if params.tree.num_leaves < 1 {
            return Err("num_leaves must be at least 1".into());
        }
        if !(params.feature_fraction > 0.0 && params.feature_fraction <= 1.0) {
            return Err("feature_fraction must be in (0, 1]".into());
        }
        let n_bins: Vec<usize> = edges.iter().map(|e| e.len() + 1).collect();
        if n_bins.iter().any(|&b| b > 256) {
            return Err("at most 255 edges (256 bins) per feature".into());
        }
        if let Some(&too_big) = train
            .bins
            .iter()
            .enumerate()
            .find_map(|(i, b)| ((*b as usize) >= n_bins[i % train.d]).then_some(b))
        {
            return Err(format!(
                "bin index {too_big} is out of range for its feature"
            ));
        }
        let layout = HistLayout::new(&n_bins);
        let inv_max_dcg = inverse_max_dcg(
            &train.labels,
            &train.offsets,
            params.lambda.truncation_level,
        );
        let gains = gain_table(&train.labels);
        let longest = train
            .offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0);
        let discounts = discount_table(longest);
        let n = train.n;
        let nv = valid.as_ref().map_or(0, |v| v.n);
        Ok(Self {
            params,
            layout,
            edges,
            trees: Vec::new(),
            train_scores: vec![0.0; n],
            valid_scores: vec![0.0; nv],
            timings: Timings::default(),
            workspace: Workspace::default(),
            columns: transpose(&train.bins, train.n, train.d),
            inv_max_dcg,
            gains,
            discounts,
            grad: vec![0.0; n],
            hess: vec![0.0; n],
            indices: Vec::with_capacity(n),
            rng: SplitMix64::new(params.seed),
            train,
            valid,
        })
    }

    fn compute_gradients(&mut self) {
        match self.params.objective {
            Objective::LambdaRank => all_gradients(
                &self.train.labels,
                &self.gains,
                &self.train_scores,
                &self.train.offsets,
                &self.inv_max_dcg,
                &self.discounts,
                &self.params.lambda,
                &mut self.grad,
                &mut self.hess,
            ),
            Objective::Regression => {
                self.grad
                    .par_iter_mut()
                    .zip(self.hess.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (g, h))| {
                        *g = self.train_scores[i] - self.train.labels[i] as f64;
                        *h = 1.0;
                    });
            }
        }
    }

    /// Add one tree; return the validation NDCG@eval_at after it (if validation data exists).
    pub fn step(&mut self) -> Option<f64> {
        let clock = Instant::now();
        self.compute_gradients();
        self.timings.gradients += clock.elapsed().as_secs_f64();
        let mask = feature_mask(self.train.d, self.params.feature_fraction, &mut self.rng);
        self.indices.clear();
        self.indices.extend(0..self.train.n as u32);
        let m = Matrix {
            bins: &self.train.bins,
            columns: Some(&self.columns),
            n: self.train.n,
            d: self.train.d,
            layout: &self.layout,
            edges: &self.edges,
        };
        let (tree, ranges) = grow_tree(
            &m,
            &self.grad,
            &self.hess,
            &mut self.indices,
            &mask,
            &self.params.tree,
            self.params.learning_rate,
            &mut self.timings,
            &mut self.workspace,
        );
        let clock = Instant::now();
        for (leaf, &(s, e)) in ranges.iter().enumerate() {
            let v = tree.leaf_value[leaf];
            for &r in &self.indices[s..e] {
                self.train_scores[r as usize] += v;
            }
        }
        self.timings.update += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let result = self.valid.as_ref().map(|v| {
            let d = v.d;
            self.valid_scores
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(c, out)| {
                    let base = c * 4096;
                    for (i, o) in out.iter_mut().enumerate() {
                        let r = base + i;
                        *o += tree.predict_binned(&v.bins[r * d..(r + 1) * d]);
                    }
                });
            mean_ndcg(
                &v.labels,
                &self.valid_scores,
                &v.offsets,
                self.params.eval_at,
            )
        });
        self.timings.validation += clock.elapsed().as_secs_f64();
        self.trees.push(tree);
        result
    }

    /// Mean NDCG@eval_at of the current model on the training data.
    pub fn train_ndcg(&self) -> f64 {
        mean_ndcg(
            &self.train.labels,
            &self.train_scores,
            &self.train.offsets,
            self.params.eval_at,
        )
    }

    /// Keep only the first `k` trees (after early stopping).
    pub fn truncate(&mut self, k: usize) {
        self.trees.truncate(k);
    }
}

/// Raw-feature prediction of an ensemble for a row-major `n x d` matrix, in parallel.
pub fn predict_raw(trees: &[Tree], x: &[f32], d: usize, out: &mut [f64]) {
    out.par_chunks_mut(1024).enumerate().for_each(|(c, o)| {
        for (i, v) in o.iter_mut().enumerate() {
            let r = c * 1024 + i;
            let row = &x[r * d..(r + 1) * d];
            *v = trees.iter().map(|t| t.predict_raw(row)).sum();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy(n_queries: usize, docs: usize) -> (Binned, Vec<Vec<f32>>) {
        // Label = feature 0 bin (0..4); feature 1 is noise.
        let mut bins = Vec::new();
        let mut labels = Vec::new();
        let mut offsets = vec![0];
        let mut rng = SplitMix64::new(7);
        for _ in 0..n_queries {
            for _ in 0..docs {
                let a = rng.below(5) as u8;
                bins.push(a);
                bins.push(rng.below(8) as u8);
                labels.push(a as f32);
            }
            offsets.push(labels.len());
        }
        let n = labels.len();
        let edges = vec![
            (0..4).map(|v| v as f32).collect(),
            (0..7).map(|v| v as f32).collect(),
        ];
        (
            Binned {
                bins,
                n,
                d: 2,
                labels,
                offsets,
            },
            edges,
        )
    }

    #[test]
    fn lambdamart_learns_a_perfect_ranking() {
        let (train, edges) = toy(50, 20);
        let valid = train.clone();
        let params = BoostParams {
            tree: TreeParams {
                num_leaves: 7,
                min_data_in_leaf: 5,
                ..TreeParams::default()
            },
            ..BoostParams::default()
        };
        let mut t = Trainer::new(params, train, Some(valid), edges).unwrap();
        let mut last = 0.0;
        for _ in 0..20 {
            last = t.step().unwrap();
        }
        assert!(last > 0.999, "validation NDCG {last}");
        assert!(t.train_ndcg() > 0.999);
        let x: Vec<f32> = vec![0.0, 3.0, 4.0, 3.0];
        let mut out = vec![0.0; 2];
        predict_raw(&t.trees, &x, 2, &mut out);
        assert!(out[1] > out[0]);
        t.truncate(3);
        assert_eq!(t.trees.len(), 3);
    }

    #[test]
    fn regression_objective_fits_labels() {
        let (train, edges) = toy(20, 10);
        let params = BoostParams {
            objective: Objective::Regression,
            learning_rate: 0.5,
            feature_fraction: 0.5,
            ..BoostParams::default()
        };
        let mut t = Trainer::new(params, train, None, edges).unwrap();
        for _ in 0..30 {
            assert!(t.step().is_none());
        }
        assert!(t.train_ndcg() > 0.99);
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        let (mut train, edges) = toy(2, 3);
        let p = BoostParams::default();
        assert!(Trainer::new(p, train.clone(), None, edges[..1].to_vec()).is_err());
        let bad = BoostParams {
            feature_fraction: 0.0,
            ..BoostParams::default()
        };
        assert!(Trainer::new(bad, train.clone(), None, edges.clone()).is_err());
        train.bins[0] = 200;
        assert!(Trainer::new(p, train.clone(), None, edges.clone()).is_err());
        train.offsets = vec![0, 2];
        assert!(Trainer::new(p, train, None, edges).is_err());
    }

    #[test]
    fn feature_mask_is_reproducible() {
        let a = feature_mask(10, 0.3, &mut SplitMix64::new(1));
        let b = feature_mask(10, 0.3, &mut SplitMix64::new(1));
        assert_eq!(a, b);
        assert_eq!(a.iter().filter(|&&v| v).count(), 3);
        assert!(
            feature_mask(4, 1.0, &mut SplitMix64::new(0))
                .iter()
                .all(|&v| v)
        );
    }
}
