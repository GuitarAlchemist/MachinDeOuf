//! Analyze Signals with FFT
//!
//! Decompose a signal into frequency components.
//!
//! ```bash
//! cargo run --example fft_analysis
//! ```

use machin_signal::fft::{fft, power_spectrum};
use machin_signal::window::hamming;

fn main() {
    // Generate a signal: 10Hz + 50Hz components
    let sample_rate = 256.0;
    let signal: Vec<f64> = (0..256)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 10.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
        })
        .collect();

    println!("Signal: {} samples at {}Hz", signal.len(), sample_rate);
    println!("Components: 10Hz (amplitude 1.0) + 50Hz (amplitude 0.5)");

    // Apply window to reduce spectral leakage
    let windowed = hamming(&signal);

    // FFT -> frequency domain
    let spectrum = fft(&windowed);
    let power = power_spectrum(&spectrum);

    // Show top frequency bins
    let mut indexed: Vec<(usize, f64)> = power.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop frequency bins:");
    for (bin, mag) in indexed.iter().take(5) {
        let freq = *bin as f64 * sample_rate / signal.len() as f64;
        println!("  Bin {}: {:.1}Hz, power={:.4}", bin, freq, mag);
    }
}
