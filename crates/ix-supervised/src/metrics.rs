//! Evaluation metrics for supervised learning.

use ndarray::Array1;

/// Mean Squared Error.
pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|v| v * v).mean().unwrap()
}

/// Root Mean Squared Error.
pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// R² (coefficient of determination).
pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap();
    let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot < 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Classification accuracy.
pub fn accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64 {
    let correct: usize = y_true.iter().zip(y_pred.iter()).filter(|(t, p)| t == p).count();
    correct as f64 / y_true.len() as f64
}

/// Precision for a specific class.
pub fn precision(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let tp: usize = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p == class).count();
    let fp: usize = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t != class && p == class).count();
    if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 }
}

/// Recall for a specific class.
pub fn recall(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let tp: usize = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p == class).count();
    let r#fn: usize = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == class && p != class).count();
    if tp + r#fn == 0 { 0.0 } else { tp as f64 / (tp + r#fn) as f64 }
}

/// F1 score for a specific class.
pub fn f1_score(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64 {
    let p = precision(y_true, y_pred, class);
    let r = recall(y_true, y_pred, class);
    if p + r < 1e-12 { 0.0 } else { 2.0 * p * r / (p + r) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mse(&y_true, &y_pred).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!((r_squared(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy() {
        let y_true = array![0, 1, 1, 0];
        let y_pred = array![0, 1, 0, 0];
        assert!((accuracy(&y_true, &y_pred) - 0.75).abs() < 1e-10);
    }
}
