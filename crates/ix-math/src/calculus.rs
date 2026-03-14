//! Numerical calculus: gradients, differentiation.

use ndarray::Array1;

/// Numerical gradient via central finite differences.
/// f: objective function, x: point, epsilon: step size.
pub fn numerical_gradient<F>(f: &F, x: &Array1<f64>, epsilon: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    for i in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += epsilon;
        x_minus[i] -= epsilon;
        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
    }
    grad
}

/// Numerical Hessian via finite differences of the gradient.
pub fn numerical_hessian<F>(f: &F, x: &Array1<f64>, epsilon: f64) -> ndarray::Array2<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut hess = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut xpp = x.clone();
            let mut xpm = x.clone();
            let mut xmp = x.clone();
            let mut xmm = x.clone();

            xpp[i] += epsilon;
            xpp[j] += epsilon;
            xpm[i] += epsilon;
            xpm[j] -= epsilon;
            xmp[i] -= epsilon;
            xmp[j] += epsilon;
            xmm[i] -= epsilon;
            xmm[j] -= epsilon;

            hess[[i, j]] = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * epsilon * epsilon);
        }
    }
    hess
}

/// Numerical derivative of a scalar function f: R -> R.
pub fn derivative<F>(f: &F, x: f64, epsilon: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + epsilon) - f(x - epsilon)) / (2.0 * epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_numerical_gradient_quadratic() {
        // f(x) = x0^2 + x1^2, grad = [2*x0, 2*x1]
        let f = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
        let x = array![3.0, 4.0];
        let grad = numerical_gradient(&f, &x, 1e-5);
        assert!((grad[0] - 6.0).abs() < 1e-5);
        assert!((grad[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_derivative_cubic() {
        // f(x) = x^3, f'(x) = 3x^2
        let f = |x: f64| x.powi(3);
        let d = derivative(&f, 2.0, 1e-5);
        assert!((d - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_hessian_quadratic() {
        // f(x) = x0^2 + 2*x1^2, hessian = [[2, 0], [0, 4]]
        let f = |x: &Array1<f64>| x[0].powi(2) + 2.0 * x[1].powi(2);
        let x = array![1.0, 1.0];
        let h = numerical_hessian(&f, &x, 1e-4);
        assert!((h[[0, 0]] - 2.0).abs() < 1e-3);
        assert!((h[[1, 1]] - 4.0).abs() < 1e-3);
        assert!(h[[0, 1]].abs() < 1e-3);
    }
}
