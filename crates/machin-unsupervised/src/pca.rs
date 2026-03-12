//! Principal Component Analysis (PCA).
//!
//! TODO: Implement PCA via covariance eigendecomposition.

use ndarray::Array2;
use crate::traits::DimensionReducer;

pub struct PCA {
    pub n_components: usize,
    // TODO: components, explained_variance
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

impl DimensionReducer for PCA {
    fn fit(&mut self, _x: &Array2<f64>) {
        todo!("PCA fitting via eigendecomposition")
    }

    fn transform(&self, _x: &Array2<f64>) -> Array2<f64> {
        todo!("PCA transform")
    }
}
