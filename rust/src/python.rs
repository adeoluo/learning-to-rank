//! PyO3 bindings, built by maturin into the `learning_to_rank_rs` extension module.
//!
//! Arrays cross the boundary through the buffer protocol: inputs are read in place (they must be
//! C-contiguous numpy arrays of the documented dtype) and large outputs are returned as bytes for
//! `numpy.frombuffer`. The GIL is released around every heavy loop.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::binning;
use crate::boost::{self, Binned, BoostParams, Objective, Trainer};
use crate::lambda::{LambdaParams, gradients_once};
use crate::metrics;
use crate::svmlight;
use crate::tree::{Tree, TreeParams};

fn slice<'a, T: Element>(buf: &'a PyBuffer<T>, name: &str) -> PyResult<&'a [T]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    // SAFETY: the buffer is C-contiguous with element type T (checked by PyBuffer::get) and stays
    // alive (and exported, so numpy cannot resize it) for as long as `buf` is borrowed.
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const T, buf.item_count()) })
}

#[allow(clippy::mut_from_ref)]
fn slice_mut<'a, T: Element>(buf: &'a PyBuffer<T>, name: &str) -> PyResult<&'a mut [T]> {
    if buf.readonly() {
        return Err(PyValueError::new_err(format!("{name} must be writable")));
    }
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    // SAFETY: as in `slice`, and the buffer is writable; callers pass a distinct output array.
    Ok(unsafe { std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut T, buf.item_count()) })
}

fn bytes_of<'py, T: Copy>(py: Python<'py>, values: &[T]) -> Bound<'py, PyBytes> {
    // SAFETY: plain-old-data numeric slices viewed as bytes.
    let raw = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    PyBytes::new(py, raw)
}

fn offsets_usize(offsets: &[i64]) -> PyResult<Vec<usize>> {
    offsets
        .iter()
        .map(|&o| usize::try_from(o).map_err(|_| PyValueError::new_err("negative offset")))
        .collect()
}

type Parsed<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    usize,
);

/// Parse LETOR text; returns (labels f32 bytes, qids i64 bytes, features f32 bytes, n_features).
#[pyfunction]
#[pyo3(signature = (text, n_features=0))]
fn parse_svmlight<'py>(py: Python<'py>, text: &str, n_features: usize) -> PyResult<Parsed<'py>> {
    let parsed = py
        .detach(|| svmlight::parse(text, n_features))
        .map_err(PyValueError::new_err)?;
    Ok((
        bytes_of(py, &parsed.labels),
        bytes_of(py, &parsed.qids),
        bytes_of(py, &parsed.features),
        parsed.n_features,
    ))
}

/// Fit per-feature bin edges from the sample rows of a row-major float32 matrix.
#[pyfunction]
fn fit_bin_edges(
    py: Python<'_>,
    x: PyBuffer<f32>,
    d: usize,
    sample: PyBuffer<u32>,
    max_bin: usize,
) -> PyResult<Vec<Vec<f32>>> {
    let xs = slice(&x, "x")?;
    let rows = slice(&sample, "sample")?;
    if !(2..=256).contains(&max_bin) {
        return Err(PyValueError::new_err("max_bin must be in [2, 256]"));
    }
    if d == 0 || xs.len() % d != 0 {
        return Err(PyValueError::new_err("x size is not a multiple of d"));
    }
    let n = xs.len() / d;
    if rows.iter().any(|&r| r as usize >= n) {
        return Err(PyValueError::new_err("sample row out of range"));
    }
    Ok(py.detach(|| binning::fit_all(xs, d, rows, max_bin)))
}

/// Map a row-major float32 matrix to bins, writing into `out` (uint8, same shape).
#[pyfunction]
fn apply_bins(
    py: Python<'_>,
    x: PyBuffer<f32>,
    d: usize,
    edges: Vec<Vec<f32>>,
    out: PyBuffer<u8>,
) -> PyResult<()> {
    let xs = slice(&x, "x")?;
    let o = slice_mut(&out, "out")?;
    if edges.len() != d || xs.len() != o.len() || (d > 0 && xs.len() % d != 0) {
        return Err(PyValueError::new_err(
            "shape mismatch between x, edges and out",
        ));
    }
    if edges.iter().any(|e| e.len() > 255) {
        return Err(PyValueError::new_err("at most 255 edges per feature"));
    }
    py.detach(|| binning::apply(xs, d, &edges, o));
    Ok(())
}

/// LambdaRank gradients and hessians for every query (float64 bytes each).
#[pyfunction]
#[pyo3(signature = (labels, scores, offsets, truncation_level=30, sigma=1.0, norm=true))]
fn lambdarank_gradients<'py>(
    py: Python<'py>,
    labels: PyBuffer<f32>,
    scores: PyBuffer<f64>,
    offsets: PyBuffer<i64>,
    truncation_level: usize,
    sigma: f64,
    norm: bool,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let y = slice(&labels, "labels")?;
    let s = slice(&scores, "scores")?;
    let off = offsets_usize(slice(&offsets, "offsets")?)?;
    if y.len() != s.len() || off.first() != Some(&0) || off.last() != Some(&y.len()) {
        return Err(PyValueError::new_err("labels, scores and offsets disagree"));
    }
    let p = LambdaParams {
        truncation_level,
        sigma,
        norm,
    };
    let (g, h) = py.detach(|| {
        let mut g = vec![0.0; y.len()];
        let mut h = vec![0.0; y.len()];
        gradients_once(y, s, &off, &p, &mut g, &mut h);
        (g, h)
    });
    Ok((bytes_of(py, &g), bytes_of(py, &h)))
}

/// Mean NDCG@k (gain 2^label - 1, NDCG = 1 for queries without relevant documents).
#[pyfunction]
fn mean_ndcg(
    py: Python<'_>,
    labels: PyBuffer<f32>,
    scores: PyBuffer<f64>,
    offsets: PyBuffer<i64>,
    k: usize,
) -> PyResult<f64> {
    let y = slice(&labels, "labels")?;
    let s = slice(&scores, "scores")?;
    let off = offsets_usize(slice(&offsets, "offsets")?)?;
    if y.len() != s.len() || off.first() != Some(&0) || off.last() != Some(&y.len()) {
        return Err(PyValueError::new_err("labels, scores and offsets disagree"));
    }
    Ok(py.detach(|| metrics::mean_ndcg(y, s, &off, k)))
}

#[allow(clippy::too_many_arguments)]
fn trees_from_flat(
    node_offsets: &[i64],
    split_feature: &[i32],
    threshold: &[f32],
    left: &[i32],
    right: &[i32],
    leaf_offsets: &[i64],
    leaf_value: &[f64],
    d: usize,
) -> PyResult<Vec<Tree>> {
    let bad = || PyValueError::new_err("inconsistent flat tree arrays");
    if node_offsets.len() != leaf_offsets.len() || node_offsets.is_empty() {
        return Err(bad());
    }
    let nn = split_feature.len();
    if threshold.len() != nn || left.len() != nn || right.len() != nn {
        return Err(bad());
    }
    let nodes = offsets_usize(node_offsets)?;
    let leaves = offsets_usize(leaf_offsets)?;
    if nodes.last() != Some(&nn) || leaves.last() != Some(&leaf_value.len()) {
        return Err(bad());
    }
    let mut trees = Vec::with_capacity(nodes.len() - 1);
    for t in 0..nodes.len() - 1 {
        let (a, b) = (nodes[t], nodes[t + 1]);
        let (la, lb) = (leaves[t], leaves[t + 1]);
        if b < a || lb <= la || lb - la != b - a + 1 {
            return Err(bad());
        }
        let n_nodes = (b - a) as i32;
        let n_leaves = (lb - la) as i32;
        for i in a..b {
            for &c in [left[i], right[i]].iter() {
                // Children come after their parent, which also rules out cycles.
                let local = (i - a) as i32;
                if c >= n_nodes || (c < 0 && !c >= n_leaves) || (c >= 0 && c <= local) {
                    return Err(bad());
                }
            }
            if split_feature[i] < 0 || split_feature[i] as usize >= d {
                return Err(PyValueError::new_err("split feature out of range"));
            }
        }
        trees.push(Tree {
            split_feature: split_feature[a..b].iter().map(|&f| f as u32).collect(),
            threshold_bin: vec![0; b - a],
            threshold: threshold[a..b].to_vec(),
            left: left[a..b].to_vec(),
            right: right[a..b].to_vec(),
            split_gain: vec![0.0; b - a],
            leaf_value: leaf_value[la..lb].to_vec(),
            leaf_count: vec![0; lb - la],
        });
    }
    Ok(trees)
}

/// Predict a row-major float32 matrix with an ensemble given as flat arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn predict<'py>(
    py: Python<'py>,
    x: PyBuffer<f32>,
    d: usize,
    node_offsets: PyBuffer<i64>,
    split_feature: PyBuffer<i32>,
    threshold: PyBuffer<f32>,
    left: PyBuffer<i32>,
    right: PyBuffer<i32>,
    leaf_offsets: PyBuffer<i64>,
    leaf_value: PyBuffer<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let xs = slice(&x, "x")?;
    if d == 0 || xs.len() % d != 0 {
        return Err(PyValueError::new_err("x size is not a multiple of d"));
    }
    let trees = trees_from_flat(
        slice(&node_offsets, "node_offsets")?,
        slice(&split_feature, "split_feature")?,
        slice(&threshold, "threshold")?,
        slice(&left, "left")?,
        slice(&right, "right")?,
        slice(&leaf_offsets, "leaf_offsets")?,
        slice(&leaf_value, "leaf_value")?,
        d,
    )?;
    let n = xs.len() / d;
    let out = py.detach(|| {
        let mut out = vec![0.0f64; n];
        boost::predict_raw(&trees, xs, d, &mut out);
        out
    });
    Ok(bytes_of(py, &out))
}

fn binned(
    bins: &PyBuffer<u8>,
    d: usize,
    labels: &PyBuffer<f32>,
    offsets: &PyBuffer<i64>,
) -> PyResult<Binned> {
    let b = slice(bins, "bins")?;
    if d == 0 || b.len() % d != 0 {
        return Err(PyValueError::new_err("bins size is not a multiple of d"));
    }
    Ok(Binned {
        bins: b.to_vec(),
        n: b.len() / d,
        d,
        labels: slice(labels, "labels")?.to_vec(),
        offsets: offsets_usize(slice(offsets, "offsets")?)?,
    })
}

/// A boosting run held in Rust: `step(k)` adds k trees and returns the validation NDCG after each.
#[pyclass(module = "learning_to_rank_rs")]
struct Booster {
    inner: Trainer,
}

#[pymethods]
impl Booster {
    #[new]
    #[pyo3(signature = (
        bins, d, labels, offsets, edges, *, objective="lambdarank", learning_rate=0.1,
        num_leaves=31, max_depth=0, min_data_in_leaf=20, min_sum_hessian_in_leaf=1e-3,
        lambda_l2=0.0, min_gain_to_split=0.0, feature_fraction=1.0, truncation_level=30,
        sigma=1.0, norm=true, eval_at=10, seed=0, valid_bins=None, valid_labels=None,
        valid_offsets=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bins: PyBuffer<u8>,
        d: usize,
        labels: PyBuffer<f32>,
        offsets: PyBuffer<i64>,
        edges: Vec<Vec<f32>>,
        objective: &str,
        learning_rate: f64,
        num_leaves: usize,
        max_depth: usize,
        min_data_in_leaf: u64,
        min_sum_hessian_in_leaf: f64,
        lambda_l2: f64,
        min_gain_to_split: f64,
        feature_fraction: f64,
        truncation_level: usize,
        sigma: f64,
        norm: bool,
        eval_at: usize,
        seed: u64,
        valid_bins: Option<PyBuffer<u8>>,
        valid_labels: Option<PyBuffer<f32>>,
        valid_offsets: Option<PyBuffer<i64>>,
    ) -> PyResult<Self> {
        let objective = match objective {
            "lambdarank" => Objective::LambdaRank,
            "regression" => Objective::Regression,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown objective {other:?}; expected 'lambdarank' or 'regression'"
                )));
            }
        };
        let params = BoostParams {
            objective,
            learning_rate,
            tree: TreeParams {
                num_leaves,
                max_depth,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l2,
                min_gain_to_split,
            },
            lambda: LambdaParams {
                truncation_level,
                sigma,
                norm,
            },
            feature_fraction,
            seed,
            eval_at,
        };
        let train = binned(&bins, d, &labels, &offsets)?;
        let valid = match (valid_bins, valid_labels, valid_offsets) {
            (Some(b), Some(l), Some(o)) => Some(binned(&b, d, &l, &o)?),
            (None, None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "valid_bins, valid_labels and valid_offsets go together",
                ));
            }
        };
        let inner = Trainer::new(params, train, valid, edges).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Add `k` trees; returns the validation NDCG after each (None without validation data).
    fn step(&mut self, py: Python<'_>, k: usize) -> Vec<Option<f64>> {
        let inner = &mut self.inner;
        py.detach(|| (0..k).map(|_| inner.step()).collect())
    }

    fn train_ndcg(&self, py: Python<'_>) -> f64 {
        let inner = &self.inner;
        py.detach(|| inner.train_ndcg())
    }

    /// Seconds spent per phase so far: gradients, histograms, splits, partition, update, validation.
    fn timings(&self) -> Vec<(&'static str, f64)> {
        let t = self.inner.timings;
        vec![
            ("gradients", t.gradients),
            ("histograms", t.histograms),
            ("splits", t.splits),
            ("partition", t.partition),
            ("update", t.update),
            ("validation", t.validation),
        ]
    }

    fn truncate(&mut self, k: usize) {
        self.inner.truncate(k);
    }

    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.trees.len()
    }

    /// Training scores of the current model (float64 bytes).
    fn train_scores<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        bytes_of(py, &self.inner.train_scores)
    }

    /// Flat arrays of the ensemble, as bytes: see `learning_to_rank.gbdt.Ensemble`.
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let trees = &self.inner.trees;
        let mut node_offsets: Vec<i64> = vec![0];
        let mut leaf_offsets: Vec<i64> = vec![0];
        let mut split_feature: Vec<i32> = Vec::new();
        let mut threshold_bin: Vec<i32> = Vec::new();
        let mut threshold: Vec<f32> = Vec::new();
        let mut left: Vec<i32> = Vec::new();
        let mut right: Vec<i32> = Vec::new();
        let mut split_gain: Vec<f64> = Vec::new();
        let mut leaf_value: Vec<f64> = Vec::new();
        let mut leaf_count: Vec<i64> = Vec::new();
        for t in trees {
            split_feature.extend(t.split_feature.iter().map(|&f| f as i32));
            threshold_bin.extend(t.threshold_bin.iter().map(|&b| b as i32));
            threshold.extend_from_slice(&t.threshold);
            left.extend_from_slice(&t.left);
            right.extend_from_slice(&t.right);
            split_gain.extend_from_slice(&t.split_gain);
            leaf_value.extend_from_slice(&t.leaf_value);
            leaf_count.extend(t.leaf_count.iter().map(|&c| c as i64));
            node_offsets.push(split_feature.len() as i64);
            leaf_offsets.push(leaf_value.len() as i64);
        }
        let out = PyDict::new(py);
        out.set_item("node_offsets", bytes_of(py, &node_offsets))?;
        out.set_item("leaf_offsets", bytes_of(py, &leaf_offsets))?;
        out.set_item("split_feature", bytes_of(py, &split_feature))?;
        out.set_item("threshold_bin", bytes_of(py, &threshold_bin))?;
        out.set_item("threshold", bytes_of(py, &threshold))?;
        out.set_item("left", bytes_of(py, &left))?;
        out.set_item("right", bytes_of(py, &right))?;
        out.set_item("split_gain", bytes_of(py, &split_gain))?;
        out.set_item("leaf_value", bytes_of(py, &leaf_value))?;
        out.set_item("leaf_count", bytes_of(py, &leaf_count))?;
        Ok(out)
    }
}

/// Number of threads in the global Rayon pool (set RAYON_NUM_THREADS before import to change it).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn learning_to_rank_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_svmlight, m)?)?;
    m.add_function(wrap_pyfunction!(fit_bin_edges, m)?)?;
    m.add_function(wrap_pyfunction!(apply_bins, m)?)?;
    m.add_function(wrap_pyfunction!(lambdarank_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(mean_ndcg, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_class::<Booster>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
