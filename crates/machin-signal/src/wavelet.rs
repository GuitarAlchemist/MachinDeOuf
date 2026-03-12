//! Discrete Wavelet Transform (DWT).
//!
//! Haar wavelet (simplest) and Daubechies-4 (db4).

/// Haar wavelet forward transform (one level).
pub fn haar_forward(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len() / 2;
    let mut approx = Vec::with_capacity(n);
    let mut detail = Vec::with_capacity(n);

    let s = std::f64::consts::FRAC_1_SQRT_2;
    for i in 0..n {
        approx.push(s * (signal[2 * i] + signal[2 * i + 1]));
        detail.push(s * (signal[2 * i] - signal[2 * i + 1]));
    }

    (approx, detail)
}

/// Haar wavelet inverse transform (one level).
pub fn haar_inverse(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n = approx.len();
    let mut signal = vec![0.0; 2 * n];
    let s = std::f64::consts::FRAC_1_SQRT_2;

    for i in 0..n {
        signal[2 * i] = s * (approx[i] + detail[i]);
        signal[2 * i + 1] = s * (approx[i] - detail[i]);
    }

    signal
}

/// Multi-level Haar DWT. Returns (final_approx, [detail_at_each_level]).
pub fn haar_dwt(signal: &[f64], levels: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut current = signal.to_vec();
    let mut details = Vec::new();

    for _ in 0..levels {
        if current.len() < 2 {
            break;
        }
        let (approx, detail) = haar_forward(&current);
        details.push(detail);
        current = approx;
    }

    (current, details)
}

/// Multi-level Haar inverse DWT.
pub fn haar_idwt(approx: &[f64], details: &[Vec<f64>]) -> Vec<f64> {
    let mut current = approx.to_vec();
    for detail in details.iter().rev() {
        current = haar_inverse(&current, detail);
    }
    current
}

/// Wavelet denoising via soft thresholding.
/// Applies DWT, thresholds detail coefficients, then inverse DWT.
pub fn wavelet_denoise(signal: &[f64], levels: usize, threshold: f64) -> Vec<f64> {
    let (approx, mut details) = haar_dwt(signal, levels);

    for detail in details.iter_mut() {
        for coeff in detail.iter_mut() {
            *coeff = soft_threshold(*coeff, threshold);
        }
    }

    haar_idwt(&approx, &details)
}

fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = haar_forward(&signal);
        let recovered = haar_inverse(&approx, &detail);

        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multi_level_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, details) = haar_dwt(&signal, 3);
        let recovered = haar_idwt(&approx, &details);

        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wavelet_denoise() {
        // Clean signal + noise
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.3 * ((i * 7) as f64 % 1.3 - 0.65))
            .collect();

        let denoised = wavelet_denoise(&noisy, 3, 0.3);

        // Denoised should be closer to clean than noisy is
        let noisy_error: f64 = noisy.iter().zip(clean.iter()).map(|(n, c)| (n - c).powi(2)).sum();
        let denoised_error: f64 = denoised.iter().zip(clean.iter()).map(|(d, c)| (d - c).powi(2)).sum();

        assert!(denoised_error < noisy_error, "Denoising should reduce error");
    }
}
