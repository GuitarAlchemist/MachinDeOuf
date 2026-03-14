//! Core traits for supervised learning.

use ndarray::{Array1, Array2};

/// A regression model: predicts continuous values.
pub trait Regressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}

/// A classification model: predicts discrete labels.
pub trait Classifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
}
