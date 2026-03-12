//! Core traits for unsupervised learning.

use ndarray::{Array1, Array2};

/// A clustering algorithm.
pub trait Clusterer {
    fn fit(&mut self, x: &Array2<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn fit_predict(&mut self, x: &Array2<f64>) -> Array1<usize> {
        self.fit(x);
        self.predict(x)
    }
}

/// A dimensionality reduction algorithm.
pub trait DimensionReducer {
    fn fit(&mut self, x: &Array2<f64>);
    fn transform(&self, x: &Array2<f64>) -> Array2<f64>;
    fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}
