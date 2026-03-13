# Digital Filters (FIR and IIR)

## The Problem

You are designing an audio equaliser for a music streaming application. The raw audio
contains low-frequency hum from a power supply (60 Hz), high-frequency hiss from a noisy
microphone preamp, and the actual music you want to keep. You need to surgically remove
unwanted frequency bands without distorting the rest.

Digital filters solve this across domains: removing baseline wander from ECG signals,
anti-aliasing before downsampling, isolating a radio channel from adjacent channels, and
smoothing noisy sensor readings in real time.

## The Intuition

A digital filter is a recipe for computing each output sample as a weighted combination
of recent input samples (and sometimes recent output samples).

- **FIR filter (Finite Impulse Response):** Like a weighted moving average. The output
  depends only on the current and past *input* values. It always settles back to zero
  after the input stops -- the "impulse response" is finite.

- **IIR filter (Infinite Impulse Response):** The output also depends on past *output*
  values (feedback). This makes it possible to build sharper filters with fewer
  coefficients, but the feedback can ring forever -- the impulse response is theoretically
  infinite.

Think of FIR as a coffee filter (one pass, no feedback) and IIR as a thermostat
(the output feeds back to influence the next output).

## How It Works

### FIR filter

```
y[n] = sum_{k=0}^{M} h[k] * x[n-k]
```

**In plain English:** Each output sample is a weighted sum of the M+1 most recent input
samples. The weights h[] are the filter coefficients, designed to pass certain frequencies
and block others.

### IIR filter

```
y[n] = sum_{k=0}^{P} b[k] * x[n-k]  -  sum_{k=1}^{Q} a[k] * y[n-k]
```

**In plain English:** Same forward sum as FIR (the b[] coefficients), plus a feedback
sum using past output values (the a[] coefficients). The feedback is what makes IIR
filters "remember" longer than their coefficient count.

### Design methods

| Filter type | Design method | MachinDeOuf API |
|-------------|--------------|-----------------|
| FIR lowpass | Windowed sinc (Hamming window) | `FirFilter::lowpass(cutoff, order)` |
| FIR highpass | Spectral inversion of lowpass | `FirFilter::highpass(cutoff, order)` |
| FIR bandpass | Difference of two lowpass filters | `FirFilter::bandpass(low, high, order)` |
| IIR 1st-order lowpass | Exponential moving average | `IirFilter::first_order_lowpass(alpha)` |
| IIR 2nd-order lowpass | Butterworth (maximally flat) | `butterworth_lowpass_2nd(cutoff)` |

## In Rust

```rust
use machin_signal::filter::{
    FirFilter, IirFilter, butterworth_lowpass_2nd,
};

// --- FIR lowpass: keep frequencies below 0.1 * Nyquist ---
let fir = FirFilter::lowpass(0.1, 32);   // cutoff=0.1, order=32
let noisy_signal: Vec<f64> = (0..256)
    .map(|i| {
        let t = i as f64 / 256.0;
        // Low-freq signal + high-freq noise
        (2.0 * std::f64::consts::PI * 5.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 100.0 * t).sin()
    })
    .collect();
let filtered = fir.apply(&noisy_signal);
// High-frequency component is attenuated after the initial transient

// --- FIR highpass: remove low-frequency drift ---
let hp = FirFilter::highpass(0.05, 64);
let stable = hp.apply(&noisy_signal);

// --- FIR bandpass: isolate a specific frequency range ---
let bp = FirFilter::bandpass(0.05, 0.15, 64);
let band_signal = bp.apply(&noisy_signal);

// --- IIR first-order lowpass (exponential smoother) ---
let iir = IirFilter::first_order_lowpass(0.1);  // alpha=0.1 = heavy smoothing
let smoothed = iir.apply(&noisy_signal);

// --- Butterworth 2nd-order lowpass (maximally flat passband) ---
let bw = butterworth_lowpass_2nd(0.1);  // normalized cutoff 0.1
let butter_filtered = bw.apply(&noisy_signal);
```

## When To Use This

| Filter | Best for | Trade-offs |
|--------|----------|------------|
| **FIR lowpass** | Clean frequency cutoff, linear phase | Needs many taps for sharp cutoff; higher latency |
| **FIR highpass** | Removing DC offset / baseline wander | Same order requirements as lowpass |
| **FIR bandpass** | Isolating a frequency band | Order must be sufficient for both edges |
| **IIR 1st-order** | Simple real-time smoothing | Gentle roll-off; not suitable for sharp filtering |
| **Butterworth** | Maximally flat passband, sharper roll-off | Phase distortion (nonlinear phase); can ring |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `cutoff` | Normalized frequency (0 to 0.5, where 0.5 = Nyquist) | cutoff = desired_frequency_hz / sample_rate |
| `order` | Number of filter taps minus 1 (FIR) | Higher order = sharper transition but more latency; start with 32-64 |
| `alpha` | IIR smoothing factor (0 to 1) | Smaller alpha = more smoothing; alpha = dt / (RC + dt) |

## Pitfalls

1. **Filter transient.** The first `order` output samples are affected by the filter
   "filling up" with data. Discard or ignore these samples for analysis.

2. **Normalized vs absolute frequency.** All cutoff values are normalized to [0, 0.5],
   where 0.5 is the Nyquist frequency (half the sample rate). To filter at 100 Hz with
   a 1000 Hz sample rate: `cutoff = 100 / 1000 = 0.1`.

3. **IIR instability.** Feedback filters can become unstable if the denominator
   coefficients a[] are poorly chosen. The Butterworth design guarantees stability, but
   hand-tuned IIR coefficients need careful pole analysis.

4. **Phase distortion.** IIR filters introduce frequency-dependent delay (nonlinear
   phase). FIR filters with symmetric coefficients have perfectly linear phase. Use FIR
   when phase matters (e.g., audio, communications).

5. **Order vs cutoff sharpness.** A low-order FIR filter has a gradual roll-off. If you
   need a sharp cutoff, increase the order -- but this increases both computation and
   latency.

## Going Further

- Apply `machin_signal::window::hamming()` to your signal before FFT analysis, or use
  these filters as pre-processing to isolate a band before spectral analysis.
- Chain filters: `FirFilter::highpass(0.02, 64)` followed by `FirFilter::lowpass(0.2, 64)`
  for a clean bandpass with independent control of each edge.
- Use `machin_signal::fft::rfft` + `machin_signal::fft::irfft` for frequency-domain
  filtering as an alternative to time-domain convolution for very long signals.
- Feed filtered signals into `machin_signal::kalman::KalmanFilter` for state estimation
  on pre-cleaned data.
