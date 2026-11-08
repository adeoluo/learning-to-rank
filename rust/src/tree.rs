//! Histogram-based regression trees grown leaf-wise (best-first), as in LightGBM.
//!
//! The learner works on a row-major `n x d` matrix of `u8` bin indices. Each leaf owns a
//! contiguous range of a shared row-index array; splitting a leaf partitions that range stably.
//! Histograms (sum of gradients, sum of hessians and count per feature bin) are built for the
//! smaller child only; the larger child's histogram is the parent's minus the smaller one.
//! Histogram building splits the rows into a fixed number of chunks that depends only on the row
//! count, sums each chunk in parallel and adds the partial histograms in chunk order, so results
//! are deterministic for any thread count. The numpy reference uses the same chunking.

use std::time::Instant;

use rayon::prelude::*;

/// Wall-clock seconds spent in each phase of training (accumulated over trees).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    pub gradients: f64,
    pub histograms: f64,
    pub splits: f64,
    pub partition: f64,
    pub update: f64,
    pub validation: f64,
}

/// Gradient statistics of one histogram bin.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bin {
    pub g: f64,
    pub h: f64,
    pub c: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    pub num_leaves: usize,
    /// 0 means no depth limit.
    pub max_depth: usize,
    pub min_data_in_leaf: u64,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            num_leaves: 31,
            max_depth: 0,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
        }
    }
}

/// A fitted tree. Child pointers `>= 0` are internal nodes; `< 0` encode leaf `!child`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tree {
    pub split_feature: Vec<u32>,
    pub threshold_bin: Vec<u8>,
    pub threshold: Vec<f32>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub split_gain: Vec<f64>,
    pub leaf_value: Vec<f64>,
    pub leaf_count: Vec<u64>,
}

impl Tree {
    #[inline]
    fn leaf_index<F: Fn(usize, u32) -> bool>(&self, go_left: F) -> usize {
        if self.split_feature.is_empty() {
            return 0;
        }
        let mut node = 0i32;
        loop {
            let n = node as usize;
            let next = if go_left(n, self.split_feature[n]) {
                self.left[n]
            } else {
                self.right[n]
            };
            if next < 0 {
                return !next as usize;
            }
            node = next;
        }
    }

    /// Output for one row of bin indices.
    #[inline]
    pub fn predict_binned(&self, row: &[u8]) -> f64 {
        let leaf = self.leaf_index(|n, f| row[f as usize] <= self.threshold_bin[n]);
        self.leaf_value[leaf]
    }

    /// Output for one row of raw features; NaN goes left, matching bin 0.
    #[inline]
    pub fn predict_raw(&self, row: &[f32]) -> f64 {
        let leaf = self.leaf_index(|n, f| {
            let x = row[f as usize];
            x.is_nan() || x <= self.threshold[n]
        });
        self.leaf_value[leaf]
    }
}

/// Layout of the per-feature histograms inside one flat buffer.
#[derive(Debug, Clone)]
pub struct HistLayout {
    pub offsets: Vec<usize>,
    pub n_bins: Vec<usize>,
    pub total: usize,
}

impl HistLayout {
    pub fn new(n_bins: &[usize]) -> Self {
        let mut offsets = Vec::with_capacity(n_bins.len());
        let mut total = 0;
        for &b in n_bins {
            offsets.push(total);
            total += b;
        }
        Self {
            offsets,
            n_bins: n_bins.to_vec(),
            total,
        }
    }
}

/// Number of histogram chunks for `len` rows (a function of `len` only).
/// Leaves with fewer rows than this are built feature-parallel (no partial histograms).
pub const ROW_CHUNK_MIN: usize = 65_536;

/// Number of row chunks for a leaf of `len` rows (a function of `len` only): 1 below
/// `ROW_CHUNK_MIN`, else one chunk per 16,384 rows up to 16.
pub fn n_chunks(len: usize) -> usize {
    if len < ROW_CHUNK_MIN {
        1
    } else {
        (len / 16_384).clamp(1, 16)
    }
}

/// Feature-parallel accumulation: each task owns a block of features and walks all rows, so
/// every bin is summed in row order, exactly as a single sequential pass would.
#[allow(clippy::too_many_arguments)]
fn accumulate_by_features(
    bins: &[u8],
    d: usize,
    layout: &HistLayout,
    rows: &[u32],
    g: &[f64],
    h: &[f64],
    out: &mut [Bin],
) {
    const BLOCK: usize = 8;
    let mut slices: Vec<(usize, &mut [Bin])> = Vec::new();
    let mut rest: &mut [Bin] = out;
    let mut f0 = 0;
    while f0 < d {
        let f1 = (f0 + BLOCK).min(d);
        let width: usize = layout.n_bins[f0..f1].iter().sum();
        let (head, tail) = rest.split_at_mut(width);
        slices.push((f0, head));
        rest = tail;
        f0 = f1;
    }
    slices.into_par_iter().for_each(|(f0, hist)| {
        let f1 = (f0 + BLOCK).min(d);
        let base = layout.offsets[f0];
        for &r in rows {
            let r = r as usize;
            let (gr, hr) = (g[r], h[r]);
            let row = &bins[r * d + f0..r * d + f1];
            for (j, &b) in row.iter().enumerate() {
                let cell = &mut hist[layout.offsets[f0 + j] - base + b as usize];
                cell.g += gr;
                cell.h += hr;
                cell.c += 1;
            }
        }
    });
}

fn accumulate(
    bins: &[u8],
    d: usize,
    layout: &HistLayout,
    rows: &[u32],
    g: &[f64],
    h: &[f64],
    out: &mut [Bin],
) {
    assert_eq!(
        out.len(),
        layout.total,
        "histogram buffer has the wrong size"
    );
    for &r in rows {
        let r = r as usize;
        let (gr, hr) = (g[r], h[r]);
        let row = &bins[r * d..(r + 1) * d];
        for ((&b, &off), &nb) in row.iter().zip(&layout.offsets).zip(&layout.n_bins) {
            debug_assert!((b as usize) < nb);
            let _ = nb;
            // SAFETY: every bin index is below its feature's bin count (checked when the
            // trainer is built), and offsets[j] + n_bins[j] <= total == out.len().
            let cell = unsafe { out.get_unchecked_mut(off + b as usize) };
            cell.g += gr;
            cell.h += hr;
            cell.c += 1;
        }
    }
}

/// Reusable buffers for tree growth: per-chunk partial histograms, a pool of leaf histograms and
/// partition scratch space. Reusing them avoids allocating (and page-faulting) megabytes per split.
#[derive(Debug, Default)]
pub struct Workspace {
    partials: Vec<Vec<Bin>>,
    pool: Vec<Vec<Bin>>,
    scratch: Vec<u32>,
}

impl Workspace {
    fn take(&mut self, total: usize) -> Vec<Bin> {
        match self.pool.pop() {
            Some(mut v) if v.len() == total => {
                v.iter_mut().for_each(|b| *b = Bin::default());
                v
            }
            _ => vec![Bin::default(); total],
        }
    }

    fn give(&mut self, v: Vec<Bin>) {
        if !v.is_empty() {
            self.pool.push(v);
        }
    }
}

/// Histogram of the rows `rows` into `out` (overwritten).
#[allow(clippy::too_many_arguments)]
pub fn build_histogram(
    bins: &[u8],
    d: usize,
    layout: &HistLayout,
    rows: &[u32],
    g: &[f64],
    h: &[f64],
    out: &mut [Bin],
    ws: &mut Workspace,
) {
    out.iter_mut().for_each(|b| *b = Bin::default());
    let k = n_chunks(rows.len());
    if k == 1 {
        if rows.len() < 2048 {
            accumulate(bins, d, layout, rows, g, h, out);
        } else {
            accumulate_by_features(bins, d, layout, rows, g, h, out);
        }
        return;
    }
    let len = rows.len();
    while ws.partials.len() < k {
        ws.partials.push(Vec::new());
    }
    let partials = &mut ws.partials[..k];
    partials.par_iter_mut().enumerate().for_each(|(i, local)| {
        if local.len() == layout.total {
            local.iter_mut().for_each(|b| *b = Bin::default());
        } else {
            *local = vec![Bin::default(); layout.total];
        }
        let (s, e) = (i * len / k, (i + 1) * len / k);
        accumulate(bins, d, layout, &rows[s..e], g, h, local);
    });
    for p in partials.iter() {
        for (o, v) in out.iter_mut().zip(p.iter()) {
            o.g += v.g;
            o.h += v.h;
            o.c += v.c;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    pub gain: f64,
    pub feature: usize,
    pub bin: usize,
    pub left_g: f64,
    pub left_h: f64,
    pub left_c: u64,
}

#[inline]
fn leaf_gain(g: f64, h: f64, lambda: f64) -> f64 {
    let den = h + lambda;
    if den > 0.0 { g * g / den } else { 0.0 }
}

/// Leaf output before shrinkage.
#[inline]
pub fn leaf_output(g: f64, h: f64, lambda: f64) -> f64 {
    let den = h + lambda;
    if den > 0.0 { -g / den } else { 0.0 }
}

/// Best split of one leaf over the allowed features; ties keep the lower feature and bin.
pub fn best_split(
    hist: &[Bin],
    layout: &HistLayout,
    feature_mask: &[bool],
    sum_g: f64,
    sum_h: f64,
    count: u64,
    p: &TreeParams,
) -> Option<Split> {
    let parent = leaf_gain(sum_g, sum_h, p.lambda_l2);
    let per_feature: Vec<Option<Split>> = (0..layout.n_bins.len())
        .into_par_iter()
        .map(|j| {
            if !feature_mask[j] || layout.n_bins[j] < 2 {
                return None;
            }
            let base = layout.offsets[j];
            let (mut gl, mut hl, mut cl) = (0.0f64, 0.0f64, 0u64);
            let mut best: Option<Split> = None;
            for t in 0..layout.n_bins[j] - 1 {
                let b = hist[base + t];
                gl += b.g;
                hl += b.h;
                cl += b.c;
                if cl < p.min_data_in_leaf || hl < p.min_sum_hessian_in_leaf {
                    continue;
                }
                let cr = count - cl;
                if cr < p.min_data_in_leaf {
                    break;
                }
                let (gr, hr) = (sum_g - gl, sum_h - hl);
                if hr < p.min_sum_hessian_in_leaf {
                    continue;
                }
                let gain = leaf_gain(gl, hl, p.lambda_l2) + leaf_gain(gr, hr, p.lambda_l2) - parent;
                if gain > p.min_gain_to_split && best.is_none_or(|s| gain > s.gain) {
                    best = Some(Split {
                        gain,
                        feature: j,
                        bin: t,
                        left_g: gl,
                        left_h: hl,
                        left_c: cl,
                    });
                }
            }
            best
        })
        .collect();
    let mut best: Option<Split> = None;
    for s in per_feature.into_iter().flatten() {
        if best.is_none_or(|b| s.gain > b.gain) {
            best = Some(s);
        }
    }
    best
}

struct Leaf {
    begin: usize,
    end: usize,
    g: f64,
    h: f64,
    c: u64,
    depth: usize,
    hist: Vec<Bin>,
    split: Option<Split>,
    parent: i32,
    is_left: bool,
}

/// Inputs shared by every tree of one boosting run.
pub struct Matrix<'a> {
    pub bins: &'a [u8],
    /// Optional feature-major copy of `bins` (`d x n`); makes partitioning cache-friendly.
    pub columns: Option<&'a [u8]>,
    pub n: usize,
    pub d: usize,
    pub layout: &'a HistLayout,
    pub edges: &'a [Vec<f32>],
}

/// Stable in-place partition of `rows` by `pred`; returns the number of rows where it holds.
/// Large ranges are split into chunks that are classified in parallel and then scattered to
/// their final positions, which gives the same order as the sequential pass.
pub fn stable_partition<F: Fn(u32) -> bool + Sync>(
    rows: &mut [u32],
    scratch: &mut Vec<u32>,
    pred: F,
) -> usize {
    const PAR_MIN: usize = 65_536;
    const CHUNK: usize = 16_384;
    let n = rows.len();
    if n < PAR_MIN {
        scratch.clear();
        let mut w = 0;
        for i in 0..n {
            let r = rows[i];
            if pred(r) {
                rows[w] = r;
                w += 1;
            } else {
                scratch.push(r);
            }
        }
        rows[w..].copy_from_slice(scratch);
        return w;
    }
    let lefts: Vec<usize> = rows
        .par_chunks(CHUNK)
        .map(|c| c.iter().filter(|&&r| pred(r)).count())
        .collect();
    let total_left: usize = lefts.iter().sum();
    let mut left_at = Vec::with_capacity(lefts.len());
    let mut right_at = Vec::with_capacity(lefts.len());
    let (mut l, mut r) = (0usize, total_left);
    for (i, &c) in lefts.iter().enumerate() {
        left_at.push(l);
        right_at.push(r);
        l += c;
        r += CHUNK.min(n - i * CHUNK) - c;
    }
    scratch.clear();
    scratch.resize(n, 0);
    let out = SharedSlice(scratch.as_mut_ptr());
    rows.par_chunks(CHUNK).enumerate().for_each(|(i, c)| {
        let (mut l, mut r) = (left_at[i], right_at[i]);
        for &row in c {
            // SAFETY: chunk i writes only positions left_at[i]..left_at[i]+lefts[i] and
            // right_at[i].., which are disjoint across chunks and within 0..n.
            if pred(row) {
                unsafe { *out.at(l) = row };
                l += 1;
            } else {
                unsafe { *out.at(r) = row };
                r += 1;
            }
        }
    });
    rows.copy_from_slice(scratch);
    total_left
}

#[derive(Clone, Copy)]
struct SharedSlice(*mut u32);
// SAFETY: used only for disjoint writes from parallel chunks (see `stable_partition`).
unsafe impl Send for SharedSlice {}
unsafe impl Sync for SharedSlice {}

impl SharedSlice {
    /// # Safety
    /// `i` must be in bounds and no other thread may access the same position.
    unsafe fn at(self, i: usize) -> *mut u32 {
        // SAFETY: the caller guarantees `i` is in bounds of the allocation.
        unsafe { self.0.add(i) }
    }
}

/// Feature-major copy of a row-major `n x d` bin matrix.
pub fn transpose(bins: &[u8], n: usize, d: usize) -> Vec<u8> {
    let mut cols = vec![0u8; n * d];
    if d == 0 {
        return cols;
    }
    cols.par_chunks_mut(n.max(1))
        .enumerate()
        .for_each(|(j, col)| {
            for (r, v) in col.iter_mut().enumerate() {
                *v = bins[r * d + j];
            }
        });
    cols
}

/// Grow one tree on the rows in `indices` (reordered in place). Returns the tree with leaf values
/// multiplied by `shrinkage`, and each leaf's `[begin, end)` range in `indices`.
#[allow(clippy::too_many_arguments)]
pub fn grow_tree(
    m: &Matrix<'_>,
    g: &[f64],
    h: &[f64],
    indices: &mut [u32],
    feature_mask: &[bool],
    p: &TreeParams,
    shrinkage: f64,
    timings: &mut Timings,
    ws: &mut Workspace,
) -> (Tree, Vec<(usize, usize)>) {
    let clock = Instant::now();
    let mut root_hist = ws.take(m.layout.total);
    build_histogram(m.bins, m.d, m.layout, indices, g, h, &mut root_hist, ws);
    timings.histograms += clock.elapsed().as_secs_f64();
    let (mut sg, mut sh) = (0.0f64, 0.0f64);
    for &r in indices.iter() {
        sg += g[r as usize];
        sh += h[r as usize];
    }
    let count = indices.len() as u64;
    let can_split = |depth: usize, c: u64| {
        (p.max_depth == 0 || depth < p.max_depth) && c >= 2 * p.min_data_in_leaf.max(1)
    };
    let clock = Instant::now();
    let root_split = if can_split(0, count) && p.num_leaves > 1 {
        best_split(&root_hist, m.layout, feature_mask, sg, sh, count, p)
    } else {
        None
    };
    timings.splits += clock.elapsed().as_secs_f64();
    let mut leaves = vec![Leaf {
        begin: 0,
        end: indices.len(),
        g: sg,
        h: sh,
        c: count,
        depth: 0,
        hist: root_hist,
        split: root_split,
        parent: -1,
        is_left: true,
    }];
    let mut tree = Tree::default();
    let mut scratch: Vec<u32> = std::mem::take(&mut ws.scratch);
    while leaves.len() < p.num_leaves {
        let mut chosen: Option<usize> = None;
        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(s) = leaf.split
                && chosen.is_none_or(|c| s.gain > leaves[c].split.expect("set").gain)
            {
                chosen = Some(i);
            }
        }
        let Some(li) = chosen else { break };
        let s = leaves[li].split.take().expect("chosen leaf has a split");
        let (begin, end) = (leaves[li].begin, leaves[li].end);
        // Stable partition of this leaf's rows.
        let clock = Instant::now();
        let goes_left = |r: u32| -> bool {
            match m.columns {
                Some(cols) => cols[s.feature * m.n + r as usize] as usize <= s.bin,
                None => m.bins[r as usize * m.d + s.feature] as usize <= s.bin,
            }
        };
        let w = stable_partition(&mut indices[begin..end], &mut scratch, goes_left) + begin;
        let mid = w;
        timings.partition += clock.elapsed().as_secs_f64();
        let node = tree.split_feature.len() as i32;
        if leaves[li].parent >= 0 {
            let pn = leaves[li].parent as usize;
            if leaves[li].is_left {
                tree.left[pn] = node;
            } else {
                tree.right[pn] = node;
            }
        }
        let new_leaf = leaves.len();
        tree.split_feature.push(s.feature as u32);
        tree.threshold_bin.push(s.bin as u8);
        tree.threshold.push(m.edges[s.feature][s.bin]);
        tree.left.push(!(li as i32));
        tree.right.push(!(new_leaf as i32));
        tree.split_gain.push(s.gain);

        let depth = leaves[li].depth + 1;
        let (lg, lh, lc) = (s.left_g, s.left_h, s.left_c);
        let (rg, rh, rc) = (leaves[li].g - lg, leaves[li].h - lh, leaves[li].c - lc);
        let clock = Instant::now();
        let mut parent_hist = std::mem::take(&mut leaves[li].hist);
        let mut small_hist = ws.take(m.layout.total);
        let left_smaller = (mid - begin) <= (end - mid);
        let small_rows = if left_smaller {
            &indices[begin..mid]
        } else {
            &indices[mid..end]
        };
        build_histogram(m.bins, m.d, m.layout, small_rows, g, h, &mut small_hist, ws);
        for (pv, sv) in parent_hist.iter_mut().zip(small_hist.iter()) {
            pv.g -= sv.g;
            pv.h -= sv.h;
            pv.c -= sv.c;
        }
        let (left_hist, right_hist) = if left_smaller {
            (small_hist, parent_hist)
        } else {
            (parent_hist, small_hist)
        };
        timings.histograms += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let room = leaves.len() + 1 < p.num_leaves;
        let left_split = if room && can_split(depth, lc) {
            best_split(&left_hist, m.layout, feature_mask, lg, lh, lc, p)
        } else {
            None
        };
        let right_split = if room && can_split(depth, rc) {
            best_split(&right_hist, m.layout, feature_mask, rg, rh, rc, p)
        } else {
            None
        };
        timings.splits += clock.elapsed().as_secs_f64();
        leaves[li] = Leaf {
            begin,
            end: mid,
            g: lg,
            h: lh,
            c: lc,
            depth,
            hist: left_hist,
            split: left_split,
            parent: node,
            is_left: true,
        };
        leaves.push(Leaf {
            begin: mid,
            end,
            g: rg,
            h: rh,
            c: rc,
            depth,
            hist: right_hist,
            split: right_split,
            parent: node,
            is_left: false,
        });
    }
    ws.scratch = scratch;
    let mut ranges = Vec::with_capacity(leaves.len());
    for leaf in leaves.iter_mut() {
        ws.give(std::mem::take(&mut leaf.hist));
        tree.leaf_value
            .push(leaf_output(leaf.g, leaf.h, p.lambda_l2) * shrinkage);
        tree.leaf_count.push(leaf.c);
        ranges.push((leaf.begin, leaf.end));
    }
    (tree, ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix<'a>(
        bins: &'a [u8],
        d: usize,
        layout: &'a HistLayout,
        edges: &'a [Vec<f32>],
    ) -> Matrix<'a> {
        Matrix {
            bins,
            columns: None,
            n: bins.len() / d,
            d,
            layout,
            edges,
        }
    }

    #[test]
    fn splits_on_the_informative_feature() {
        // Feature 0 separates negative from positive gradients; feature 1 is noise.
        let n = 200;
        let mut bins = Vec::new();
        let mut g = Vec::new();
        for i in 0..n {
            bins.push((i % 2) as u8);
            bins.push((i % 7) as u8);
            g.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let h = vec![1.0; n];
        let layout = HistLayout::new(&[2, 7]);
        let edges = vec![vec![0.5], (0..6).map(|v| v as f32).collect()];
        let m = matrix(&bins, 2, &layout, &edges);
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let p = TreeParams {
            num_leaves: 2,
            ..TreeParams::default()
        };
        let (tree, ranges) = grow_tree(
            &m,
            &g,
            &h,
            &mut idx,
            &[true, true],
            &p,
            0.1,
            &mut Timings::default(),
            &mut Workspace::default(),
        );
        assert_eq!(tree.split_feature, vec![0]);
        assert_eq!(tree.threshold, vec![0.5]);
        assert!((tree.leaf_value[0] + 0.1).abs() < 1e-12);
        assert!((tree.leaf_value[1] - 0.1).abs() < 1e-12);
        assert_eq!(ranges, vec![(0, 100), (100, 200)]);
        assert_eq!(tree.predict_binned(&[0, 3]), tree.leaf_value[0]);
        assert_eq!(tree.predict_raw(&[0.7, 3.0]), tree.leaf_value[1]);
        assert_eq!(tree.predict_raw(&[f32::NAN, 3.0]), tree.leaf_value[0]);
    }

    #[test]
    fn respects_min_data_and_masks() {
        let bins: Vec<u8> = (0..40).map(|i| (i % 2) as u8).collect();
        let g: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let h = vec![1.0; 40];
        let layout = HistLayout::new(&[2]);
        let edges = vec![vec![0.0]];
        let m = matrix(&bins, 1, &layout, &edges);
        let mut idx: Vec<u32> = (0..40).collect();
        let p = TreeParams {
            min_data_in_leaf: 21,
            ..TreeParams::default()
        };
        let (tree, _) = grow_tree(
            &m,
            &g,
            &h,
            &mut idx,
            &[true],
            &p,
            1.0,
            &mut Timings::default(),
            &mut Workspace::default(),
        );
        assert!(tree.split_feature.is_empty());
        assert_eq!(tree.leaf_value, vec![0.0]);
        let (tree, _) = grow_tree(
            &m,
            &g,
            &h,
            &mut idx,
            &[false],
            &TreeParams::default(),
            1.0,
            &mut Timings::default(),
            &mut Workspace::default(),
        );
        assert!(tree.split_feature.is_empty());
        assert_eq!(tree.predict_binned(&[1]), 0.0);
    }

    #[test]
    fn chunked_histogram_equals_sequential_counts() {
        let n = 50_000;
        let bins: Vec<u8> = (0..n).map(|i| (i % 13) as u8).collect();
        let g: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let h = vec![0.5; n];
        let layout = HistLayout::new(&[13]);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut out = vec![Bin::default(); 13];
        build_histogram(
            &bins,
            1,
            &layout,
            &rows,
            &g,
            &h,
            &mut out,
            &mut Workspace::default(),
        );
        assert_eq!(out.iter().map(|b| b.c).sum::<u64>(), n as u64);
        assert!((out.iter().map(|b| b.h).sum::<f64>() - 25_000.0).abs() < 1e-9);
        assert_eq!(n_chunks(1), 1);
        assert_eq!(n_chunks(65_535), 1);
        assert_eq!(n_chunks(65_536), 4);
        assert_eq!(n_chunks(1_000_000), 16);
        // Parallel and sequential partitions agree.
        for n in [10usize, 70_000, 200_001] {
            let mut a: Vec<u32> = (0..n as u32).rev().collect();
            let mut b = a.clone();
            let mut scratch = Vec::new();
            let ka = stable_partition(&mut a, &mut scratch, |r| r % 3 == 0);
            let mut left: Vec<u32> = b.iter().copied().filter(|r| r % 3 == 0).collect();
            let right: Vec<u32> = b.iter().copied().filter(|r| r % 3 != 0).collect();
            assert_eq!(ka, left.len());
            left.extend(right);
            b.copy_from_slice(&left);
            assert_eq!(a, b);
        }
        // The feature-parallel and sequential paths give identical sums.
        let rows: Vec<u32> = (0..5000).collect();
        let mut a = vec![Bin::default(); 13];
        let mut b = vec![Bin::default(); 13];
        accumulate(&bins, 1, &layout, &rows, &g, &h, &mut a);
        accumulate_by_features(&bins, 1, &layout, &rows, &g, &h, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn grows_up_to_num_leaves_with_depth_limit() {
        let n = 4000usize;
        let mut bins = Vec::new();
        let mut g = Vec::new();
        for i in 0..n {
            let a = (i % 4) as u8;
            let b = ((i / 4) % 4) as u8;
            bins.push(a);
            bins.push(b);
            g.push(a as f64 * 3.0 - b as f64 + ((i * 31) % 7) as f64 * 0.01);
        }
        let h = vec![1.0; n];
        let layout = HistLayout::new(&[4, 4]);
        let edges = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        let m = matrix(&bins, 2, &layout, &edges);
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let p = TreeParams {
            num_leaves: 8,
            ..TreeParams::default()
        };
        let (tree, ranges) = grow_tree(
            &m,
            &g,
            &h,
            &mut idx,
            &[true, true],
            &p,
            1.0,
            &mut Timings::default(),
            &mut Workspace::default(),
        );
        assert_eq!(tree.leaf_value.len(), 8);
        assert_eq!(ranges.iter().map(|(s, e)| e - s).sum::<usize>(), n);
        let p = TreeParams {
            num_leaves: 8,
            max_depth: 1,
            ..TreeParams::default()
        };
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let (tree, _) = grow_tree(
            &m,
            &g,
            &h,
            &mut idx,
            &[true, true],
            &p,
            1.0,
            &mut Timings::default(),
            &mut Workspace::default(),
        );
        assert_eq!(tree.leaf_value.len(), 2);
    }
}
