//! Decision Tree (CART) for classification.
//!
//! TODO: Implement CART with Gini impurity splitting.

use ndarray::{Array1, Array2};
use crate::traits::Classifier;

pub struct DecisionTree {
    pub max_depth: usize,
    // TODO: tree structure
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }
}

impl Classifier for DecisionTree {
    fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<usize>) {
        todo!("CART decision tree fitting")
    }

    fn predict(&self, _x: &Array2<f64>) -> Array1<usize> {
        todo!("CART decision tree prediction")
    }

    fn predict_proba(&self, _x: &Array2<f64>) -> Array2<f64> {
        todo!("CART decision tree probability estimation")
    }
}
