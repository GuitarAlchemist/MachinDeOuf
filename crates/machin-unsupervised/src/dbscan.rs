//! DBSCAN density-based clustering.
//!
//! TODO: Implement DBSCAN with epsilon-neighborhood.

use ndarray::{Array1, Array2};
use crate::traits::Clusterer;

pub struct DBSCAN {
    pub eps: f64,
    pub min_points: usize,
}

impl DBSCAN {
    pub fn new(eps: f64, min_points: usize) -> Self {
        Self { eps, min_points }
    }
}

impl Clusterer for DBSCAN {
    fn fit(&mut self, _x: &Array2<f64>) {
        todo!("DBSCAN fitting")
    }

    fn predict(&self, _x: &Array2<f64>) -> Array1<usize> {
        todo!("DBSCAN prediction")
    }
}
