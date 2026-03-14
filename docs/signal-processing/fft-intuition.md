# Fast Fourier Transform (FFT)

## The Problem

You are building a vibration monitoring system for industrial machinery. Sensors record
thousands of acceleration samples per second, but a raw time-domain waveform tells you
almost nothing about *which bearing is failing*. Each mechanical fault vibrates at a
characteristic frequency. You need to decompose the raw signal into its constituent
frequencies so you can match peaks to known fault signatures.

The same challenge appears in audio spectrum analysers, radio receivers, medical
ultrasound, and speech recognition pipelines.

## The Intuition

Imagine you are standing in a room where a pianist, a guitarist, and a drummer are all
playing simultaneously. Your ears hear a single combined waveform, yet your brain
effortlessly picks out each instrument. The FFT does the same thing mathematically: it
takes one mixed signal and decomposes it into a list of pure sine-wave ingredients, each
with its own frequency and strength.

Think of the signal as a smoothie. The FFT is a machine that un-blends the smoothie back
into its original fruits, telling you "there is 40% strawberry (10 Hz), 30% banana
(50 Hz), and 30% mango (120 Hz)."

## How It Works

### The Discrete Fourier Transform (DFT)

Given N samples x[0], x[1], ..., x[N-1], the DFT computes N complex frequency-domain
coefficients:

```
X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n/N}
```

**In plain English:** For each frequency bin k, spin through every sample, multiplying
it by a rotating complex number at that frequency. The result tells you how much of
frequency k is present (the magnitude) and where it starts in the cycle (the phase).

Direct evaluation costs O(N^2). The FFT exploits symmetry in the rotation factors
(twiddle factors) to split an N-point DFT into two N/2-point DFTs recursively,
reducing cost to O(N log N).

### Key relationships

| Formula | In plain English |
|---------|-----------------|
| `magnitude = sqrt(re^2 + im^2)` | How loud is this frequency? |
| `phase = atan2(im, re)` | Where in its cycle does this frequency start? |
| `power = re^2 + im^2` | Energy concentrated at this frequency |
| `frequency_bins[k] = k * sample_rate / N` | Which real-world frequency does bin k correspond to? |
| `ifft(fft(x)) = x` | You can perfectly reconstruct the original signal |

### Parseval's theorem

Total energy in the time domain equals total energy in the frequency domain divided by N.
This is a useful sanity check: if your FFT output has wildly different total energy,
something went wrong.

## In Rust

```rust
use ix_signal::fft::{
    Complex, fft, ifft, rfft, irfft,
    magnitude_spectrum, power_spectrum, frequency_bins,
};

// 1. Build a test signal: 10 Hz + 50 Hz at 256 Hz sample rate
let sample_rate = 256.0;
let n = 256;
let signal: Vec<f64> = (0..n)
    .map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
    })
    .collect();

// 2. Compute FFT of the real-valued signal
let spectrum = rfft(&signal);

// 3. Inspect the magnitude and power
let mags = magnitude_spectrum(&spectrum);
let power = power_spectrum(&spectrum);

// 4. Map bin indices to Hz
let freqs = frequency_bins(n, sample_rate);
for (bin, &mag) in mags.iter().enumerate().take(n / 2) {
    if mag > 10.0 {
        println!("Peak at {:.1} Hz, magnitude = {:.2}", freqs[bin], mag);
    }
}

// 5. Round-trip: reconstruct the signal from the spectrum
let recovered = irfft(&spectrum);
for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a - b).abs() < 1e-10);
}

// 6. Work with complex numbers directly
let c = Complex::new(3.0, 4.0);
assert!((c.magnitude() - 5.0).abs() < 1e-10);
assert!((c.phase() - (4.0_f64).atan2(3.0)).abs() < 1e-10);
```

> Full runnable example: [examples/signal/fft_analysis.rs](../../examples/signal/fft_analysis.rs)

## When To Use This

| Technique | Best for | Limitations |
|-----------|----------|-------------|
| **FFT** | Stationary signals, real-time spectrum analysis, frequency identification | Assumes signal is periodic over the window; poor time resolution |
| Wavelets | Non-stationary signals, transient detection | Higher computational cost; more parameters to tune |
| Short-Time FFT (STFT) | Time-varying spectra, spectrograms | Fixed window trades time vs frequency resolution |
| FIR/IIR filters | Removing known frequency bands | You already need to know which frequencies to keep/remove |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `fft_size` (N) | Frequency resolution = sample_rate / N | Larger N = finer frequency resolution but poorer time resolution |
| `sample_rate` | Maximum detectable frequency = sample_rate / 2 (Nyquist) | Must be at least 2x the highest frequency of interest |
| Window function | Reduces spectral leakage from non-periodic edges | Apply a Hamming or Hanning window before calling `fft` |

## Pitfalls

1. **Non-power-of-2 lengths.** The implementation zero-pads to the next power of 2
   automatically, but this changes your frequency bin spacing. Pad explicitly if you
   need exact bin alignment.

2. **Spectral leakage.** If your signal does not contain an exact integer number of
   cycles within the window, energy "leaks" from a true peak into neighbouring bins.
   Always apply a window function (Hamming, Hanning, Blackman) before the FFT.

3. **Aliasing.** Frequencies above sample_rate/2 fold back into lower bins, producing
   phantom peaks. Ensure your sample rate satisfies the Nyquist criterion.

4. **Interpreting the second half.** For real-valued inputs, bins N/2+1 through N-1 are
   the conjugate mirror of bins 1 through N/2-1. Only the first N/2+1 bins carry unique
   information.

5. **DC offset.** A non-zero mean in your signal creates a large spike at bin 0 (DC).
   Subtract the mean before the FFT if you only care about oscillating components.

## Going Further

- Apply a **Hamming window** before the FFT to reduce leakage:
  `ix_signal::window::hamming(n)`.
- Use `ix_signal::spectral` for **Short-Time FFT** (spectrograms) when your signal's
  frequency content changes over time.
- Combine with `ix_signal::filter::FirFilter::lowpass()` to pre-filter before
  analysis, isolating a frequency band of interest.
- For non-stationary signals (e.g., seismic data, speech), see [wavelets.md](wavelets.md).
