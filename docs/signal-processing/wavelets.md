# Wavelet Transform (Haar DWT)

## The Problem

You are designing a seismic analysis system. Earthquake signals contain sharp transient
bursts (P-waves, S-waves) embedded in slowly varying background noise. An FFT tells you
*which frequencies* are present but not *when* they occurred. You need a transform that
gives both frequency and time information simultaneously, so you can pinpoint the arrival
time of each seismic phase.

The same need appears in image compression (JPEG 2000 uses wavelets), ECG anomaly
detection, and financial time-series denoising.

## The Intuition

The FFT decomposes a signal into infinite sine waves. Wavelets use short, localised
"wave-lets" (little waves) instead. Think of the difference as:

- **FFT:** Asking "what notes are in this entire song?"
- **Wavelets:** Asking "what notes are being played *right now*, second by second?"

The Haar wavelet is the simplest possible wavelet: a step function that goes +1 then -1.
Despite its simplicity, it captures the essential idea of multi-resolution analysis.

At each level of decomposition the signal is split into:
- **Approximation coefficients:** the smooth, slowly-changing part (low frequencies).
- **Detail coefficients:** the sharp, rapidly-changing part (high frequencies).

This is like repeatedly zooming out on a photograph: at each zoom level you lose fine
detail but keep the overall structure.

## How It Works

### One-level Haar forward transform

Given a signal of length N, produce N/2 approximation and N/2 detail coefficients:

```
approx[i] = (1/sqrt(2)) * (signal[2i] + signal[2i+1])
detail[i] = (1/sqrt(2)) * (signal[2i] - signal[2i+1])
```

**In plain English:** Take each pair of adjacent samples. Their average (scaled) becomes
an approximation coefficient. Their difference (scaled) becomes a detail coefficient.
The approximation captures the trend; the detail captures the change.

### Multi-level DWT

Apply the forward transform recursively to the approximation coefficients. After L
levels you get one final approximation vector and L detail vectors, each at a different
scale (frequency band).

### Denoising via soft thresholding

```
coeff_denoised = sign(coeff) * max(|coeff| - threshold, 0)
```

**In plain English:** Small detail coefficients are likely noise, large ones are likely
signal. Soft thresholding shrinks everything toward zero by the threshold amount, and
anything below the threshold becomes exactly zero.

## In Rust

```rust
use machin_signal::wavelet::{
    haar_forward, haar_inverse,
    haar_dwt, haar_idwt,
    wavelet_denoise,
};

// --- One-level transform ---
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let (approx, detail) = haar_forward(&signal);
// approx captures the smooth trend, detail captures sharp changes

// Perfect reconstruction
let recovered = haar_inverse(&approx, &detail);
for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a - b).abs() < 1e-10);
}

// --- Multi-level decomposition ---
let (final_approx, details) = haar_dwt(&signal, 3);
// details[0] = finest scale (highest frequencies)
// details[2] = coarsest scale (lowest frequencies still above DC)
// final_approx = the very smooth residual

let reconstructed = haar_idwt(&final_approx, &details);
assert_eq!(reconstructed.len(), signal.len());

// --- Denoising a noisy signal ---
let noisy: Vec<f64> = (0..64)
    .map(|i| (i as f64 * 0.1).sin() + 0.3 * ((i * 7) as f64 % 1.3 - 0.65))
    .collect();

let clean = wavelet_denoise(&noisy, 3, 0.4);
// clean retains the sine wave but removes the pseudo-random noise
assert_eq!(clean.len(), noisy.len());
```

## When To Use This

| Technique | Best for | Limitations |
|-----------|----------|-------------|
| **Haar DWT** | Fast multi-resolution analysis, denoising, compression | Step-function basis produces blocky artefacts |
| FFT | Pure frequency analysis of stationary signals | No time localisation |
| STFT | Time-frequency analysis with fixed resolution | Resolution trade-off is fixed by window size |
| Daubechies wavelets | Smoother wavelet analysis (fewer artefacts) | More complex filter design |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `levels` | Depth of multi-resolution decomposition | Each level halves the signal length; max = log2(N) |
| `threshold` | Denoising aggressiveness | Start with sigma * sqrt(2 * ln(N)) where sigma is the noise standard deviation |
| Signal length | Must be divisible by 2^levels | Pad with zeros if needed |

## Pitfalls

1. **Signal length must be even** (and divisible by 2^levels for multi-level DWT). Odd-length
   signals will silently drop the last sample.

2. **Haar artefacts.** The Haar wavelet is a step function, so it introduces blocky
   discontinuities in denoised or compressed signals. For smoother results, consider
   higher-order wavelets.

3. **Choosing the threshold.** Too low = noise survives; too high = signal gets distorted.
   The universal threshold sigma * sqrt(2*ln(N)) is a reasonable starting point, but
   application-specific tuning is usually needed.

4. **Energy redistribution.** Unlike the FFT, wavelet coefficients at different levels
   represent different frequency bands at different time resolutions. Do not compare
   magnitudes directly across levels.

## Going Further

- Combine wavelets with `machin_signal::fft::rfft` for a two-stage analysis: wavelets
  for time localisation, FFT for precise frequency identification within each window.
- Use `machin_signal::filter::FirFilter::bandpass()` to pre-filter before wavelet analysis
  when you know the frequency band of interest.
- For image compression, apply `haar_forward` along rows, then along columns (2D DWT).
- Explore the `machin_signal::spectral` module for STFT-based time-frequency analysis
  as an alternative to wavelets.
