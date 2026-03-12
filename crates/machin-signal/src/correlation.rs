//! Auto-correlation and cross-correlation functions.

use crate::convolution;

/// Auto-correlation of a signal (normalized).
pub fn autocorrelation(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let x_rev: Vec<f64> = x.iter().rev().copied().collect();
    let raw = convolution::convolve(x, &x_rev);

    // Normalize by the zero-lag value
    let zero_lag = raw[n - 1];
    if zero_lag.abs() < 1e-15 {
        return vec![0.0; 2 * n - 1];
    }
    raw.iter().map(|&v| v / zero_lag).collect()
}

/// Cross-correlation of two signals.
pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    convolution::cross_correlate(a, b)
}

/// Normalized cross-correlation (Pearson correlation at each lag).
pub fn normalized_cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    let raw = cross_correlation(a, b);
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;

    if denom < 1e-15 {
        return vec![0.0; raw.len()];
    }
    raw.iter().map(|&v| v / denom).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_peak_at_zero_lag() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation(&signal);
        // Zero-lag should be 1.0 (normalized)
        let mid = signal.len() - 1;
        assert!((acf[mid] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_correlation_identical() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let xcorr = normalized_cross_correlation(&a, &a);
        // Should peak at zero lag
        let mid = a.len() - 1;
        assert!(xcorr[mid] > 0.99);
    }
}
