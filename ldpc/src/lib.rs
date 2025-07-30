//! LDPC Code Library - Rust port maintaining algorithmic equivalence
//!
//! This is a faithful port of Radford Neal's LDPC codes implementation,
//! adapted to use Rust idioms while preserving the exact algorithms.

pub mod dense_matrix;
pub mod distribution;
pub mod generator;
pub mod ldpc;
pub mod mod2convert;
pub mod sparse_matrix;

// Re-export main types
pub use dense_matrix::{DenseError, Mod2Dense};
pub use distribution::{Distribution, DistributionError};
pub use generator::{GeneratorMatrix, GeneratorMethod, Mod2SparseStrategy};
pub use ldpc::{LdpcCode, MakeMethod};
pub use sparse_matrix::{Mod2Entry, Mod2Sparse, SparseError};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LdpcError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Matrix dimensions mismatch")]
    DimensionMismatch,
    #[error("Distribution error: {0}")]
    Distribution(#[from] distribution::DistributionError),
    #[error("Sparse matrix error: {0}")]
    SparseMatrix(#[from] sparse_matrix::SparseError),
    #[error("Dense matrix error: {0}")]
    DenseMatrix(#[from] dense_matrix::DenseError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Encoding error: {0}")]
    Encoding(String),
    #[error("Decoding error: {0}")]
    Decoding(String),
}
