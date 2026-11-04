//! Rust core of `learning_to_rank`: LETOR parsing, feature binning, LambdaRank gradients,
//! histogram tree learning, boosting and NDCG. Every routine has a numpy reference in the Python
//! package, and `tests/test_rust_parity.py` checks agreement.

pub mod binning;
pub mod boost;
pub mod lambda;
pub mod metrics;
pub mod svmlight;
pub mod tree;

#[cfg(feature = "python")]
mod python;
