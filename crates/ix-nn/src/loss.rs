//! Loss functions for neural networks.

use ndarray::Array2;

/// Mean Squared Error loss.
pub fn mse_loss(predicted: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let diff = predicted - target;
    diff.mapv(|v| v * v).mean().unwrap()
}

/// MSE gradient: d(MSE)/d(predicted) = 2*(predicted - target)/n
pub fn mse_gradient(predicted: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    let n = predicted.nrows() as f64;
    2.0 * (predicted - target) / n
}

/// Binary cross-entropy loss.
pub fn binary_cross_entropy(predicted: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let eps = 1e-12;
    let p = predicted.mapv(|v| v.clamp(eps, 1.0 - eps));
    let loss = -(target * &p.mapv(f64::ln) + (1.0 - target) * (1.0 - &p).mapv(f64::ln));
    loss.mean().unwrap()
}

/// Binary cross-entropy gradient.
pub fn binary_cross_entropy_gradient(predicted: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-12;
    let p = predicted.mapv(|v| v.clamp(eps, 1.0 - eps));
    let n = predicted.nrows() as f64;
    (-target / &p + (1.0 - target) / (1.0 - &p)) / n
}
