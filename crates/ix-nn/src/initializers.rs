//! Weight initialization strategies.

use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

/// Xavier/Glorot initialization.
pub fn xavier(rows: usize, cols: usize) -> Array2<f64> {
    let std = (2.0 / (rows + cols) as f64).sqrt();
    Array2::random((rows, cols), Normal::new(0.0, std).unwrap())
}

/// He initialization (for ReLU networks).
pub fn he(rows: usize, cols: usize) -> Array2<f64> {
    let std = (2.0 / rows as f64).sqrt();
    Array2::random((rows, cols), Normal::new(0.0, std).unwrap())
}

/// Zero initialization.
pub fn zeros(rows: usize, cols: usize) -> Array2<f64> {
    Array2::zeros((rows, cols))
}
