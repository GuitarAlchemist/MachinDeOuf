//! Sampling, decimation, and interpolation.

use std::f64::consts::PI;

/// Downsample a signal by factor M (take every M-th sample).
pub fn decimate(signal: &[f64], factor: usize) -> Vec<f64> {
    signal.iter().step_by(factor).copied().collect()
}

/// Upsample a signal by factor L (insert L-1 zeros between samples).
pub fn upsample(signal: &[f64], factor: usize) -> Vec<f64> {
    let mut result = vec![0.0; signal.len() * factor];
    for (i, &s) in signal.iter().enumerate() {
        result[i * factor] = s;
    }
    result
}

/// Sinc interpolation: reconstruct signal at a fractional sample position.
/// Uses windowed sinc with `half_width` taps on each side.
pub fn sinc_interpolate(signal: &[f64], position: f64, half_width: usize) -> f64 {
    let mut result = 0.0;
    let n = signal.len() as i64;

    let center = position.floor() as i64;
    for k in (center - half_width as i64 + 1)..=(center + half_width as i64) {
        if k >= 0 && k < n {
            let x = position - k as f64;
            let sinc = if x.abs() < 1e-10 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            result += signal[k as usize] * sinc;
        }
    }

    result
}

/// Resample a signal to a new length using sinc interpolation.
pub fn resample(signal: &[f64], new_length: usize) -> Vec<f64> {
    let ratio = signal.len() as f64 / new_length as f64;
    (0..new_length)
        .map(|i| {
            let pos = i as f64 * ratio;
            sinc_interpolate(signal, pos, 8)
        })
        .collect()
}

/// Nyquist frequency for a given sample rate.
pub fn nyquist_frequency(sample_rate: f64) -> f64 {
    sample_rate / 2.0
}

/// Check if a frequency can be represented at the given sample rate.
pub fn is_above_nyquist(frequency: f64, sample_rate: f64) -> bool {
    frequency > nyquist_frequency(sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimate() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dec = decimate(&signal, 2);
        assert_eq!(dec, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_upsample() {
        let signal = vec![1.0, 2.0, 3.0];
        let up = upsample(&signal, 3);
        assert_eq!(up, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_resample_preserves_dc() {
        let signal = vec![5.0; 100];
        let resampled = resample(&signal, 50);
        for &v in &resampled {
            assert!((v - 5.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_nyquist() {
        assert_eq!(nyquist_frequency(44100.0), 22050.0);
        assert!(is_above_nyquist(25000.0, 44100.0));
        assert!(!is_above_nyquist(10000.0, 44100.0));
    }
}
