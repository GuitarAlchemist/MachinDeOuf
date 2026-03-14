//! Linear Regression (ordinary least squares via normal equation).

use ndarray::{Array1, Array2, Axis};

use crate::traits::Regressor;

/// Ordinary Least Squares Linear Regression.
///
/// Solves: w = (X^T X)^{-1} X^T y
pub struct LinearRegression {
    pub weights: Option<Array1<f64>>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: 0.0,
        }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Regressor for LinearRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n = x.nrows();
        // Add bias column (column of ones)
        let ones = Array2::ones((n, 1));
        let x_aug = ndarray::concatenate(Axis(1), &[x.view(), ones.view()]).unwrap();

        // Normal equation: w = (X^T X)^{-1} X^T y
        let xtx = x_aug.t().dot(&x_aug);
        let xty = x_aug.t().dot(y);

        // Solve via inverse (fine for small/medium datasets)
        let xtx_inv = ix_math::linalg::inverse(&xtx).expect("X^T X is singular");
        let w = xtx_inv.dot(&xty);

        let p = x.ncols();
        self.weights = Some(w.slice(ndarray::s![..p]).to_owned());
        self.bias = w[p];
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let w = self.weights.as_ref().expect("Model not fitted");
        x.dot(w) + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_regression_simple() {
        // y = 2*x + 1
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y);

        let w = model.weights.as_ref().unwrap();
        assert!((w[0] - 2.0).abs() < 1e-8, "weight should be ~2, got {}", w[0]);
        assert!((model.bias - 1.0).abs() < 1e-8, "bias should be ~1, got {}", model.bias);

        let pred = model.predict(&array![[6.0]]);
        assert!((pred[0] - 13.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_regression_multivariate() {
        // y = x0 + 2*x1
        let x = array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]];
        let y = array![3.0, 4.0, 5.0, 6.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y);

        let pred = model.predict(&array![[3.0, 3.0]]);
        assert!((pred[0] - 9.0).abs() < 1e-6);
    }
}
