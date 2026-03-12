//! Support Vector Machine (linear, binary classification).
//!
//! TODO: Implement linear SVM with subgradient descent.

use ndarray::{Array1, Array2};
use crate::traits::Classifier;

pub struct LinearSVM {
    pub c: f64, // Regularization parameter
    // TODO: weights, bias
}

impl LinearSVM {
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl Classifier for LinearSVM {
    fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<usize>) {
        todo!("Linear SVM fitting")
    }

    fn predict(&self, _x: &Array2<f64>) -> Array1<usize> {
        todo!("Linear SVM prediction")
    }

    fn predict_proba(&self, _x: &Array2<f64>) -> Array2<f64> {
        todo!("Linear SVM probability estimation")
    }
}
