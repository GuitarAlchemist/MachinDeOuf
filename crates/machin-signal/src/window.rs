//! Window functions for spectral analysis and filter design.

use std::f64::consts::PI;

/// Rectangular window (no windowing).
pub fn rectangular(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Hanning (Hann) window.
pub fn hanning(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Hamming window.
pub fn hamming(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Blackman window.
pub fn blackman(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / (n - 1) as f64;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Bartlett (triangular) window.
pub fn bartlett(n: usize) -> Vec<f64> {
    let half = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|i| 1.0 - ((i as f64 - half) / half).abs())
        .collect()
}

/// Kaiser window with parameter beta.
/// Higher beta = narrower main lobe, lower sidelobes.
pub fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            let arg = beta * (1.0 - t * t).max(0.0).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Gaussian window.
pub fn gaussian(n: usize, sigma: f64) -> Vec<f64> {
    let center = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|i| {
            let t = (i as f64 - center) / (sigma * center);
            (-0.5 * t * t).exp()
        })
        .collect()
}

/// Apply a window to a signal (element-wise multiplication).
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal
        .iter()
        .zip(window.iter())
        .map(|(s, w)| s * w)
        .collect()
}

/// Zeroth-order modified Bessel function of the first kind (for Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;

    for k in 1..50 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hanning_endpoints_zero() {
        let w = hanning(8);
        assert!(w[0].abs() < 1e-10);
        assert!(w[7].abs() < 1e-10);
    }

    #[test]
    fn test_hamming_endpoints_nonzero() {
        let w = hamming(8);
        assert!(w[0] > 0.07); // Hamming doesn't go to zero
    }

    #[test]
    fn test_rectangular_all_ones() {
        let w = rectangular(5);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_kaiser_symmetric() {
        let w = kaiser(8, 5.0);
        for i in 0..4 {
            assert!((w[i] - w[7 - i]).abs() < 1e-10);
        }
    }
}
