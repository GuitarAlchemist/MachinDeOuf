# Kalman Filter

## The Problem

You are building a GPS-based drone tracking system. The GPS receiver reports position
every second, but each reading has 3--5 metres of random error. Meanwhile, you know the
drone's velocity from its flight controller. You need to combine these two imperfect
information sources to produce a smooth, accurate position estimate -- ideally one that
also predicts where the drone will be *between* GPS updates.

Kalman filters appear everywhere noisy sensors meet a known physical model: autonomous
vehicle localisation, spacecraft navigation, financial time-series smoothing, and robotic
arm control.

## The Intuition

Imagine you are trying to figure out where a friend is walking in a foggy park. You have
two sources of information:

1. **A physical model:** "My friend walks north at 1 metre per second." This lets you
   *predict* their position, but the prediction drifts over time because you do not know
   the exact speed.
2. **A noisy sensor:** Every few seconds you glimpse them through the fog. The sighting
   is roughly correct but imprecise.

The Kalman filter is an optimal way to blend these two sources. When the sensor reading
arrives, the filter asks: "How much should I trust this measurement versus my prediction?"
It answers with the **Kalman gain** -- a number between 0 (ignore the measurement
entirely) and 1 (trust the measurement completely). Over time the filter converges to a
better estimate than either source alone.

## How It Works

### State model

```
x_k = F * x_{k-1} + B * u_k + w_k      (state transition)
z_k = H * x_k + v_k                      (observation)
```

Where w ~ N(0, Q) is process noise and v ~ N(0, R) is measurement noise.

### Predict step

```
x_predicted = F * x + B * u
P_predicted = F * P * F^T + Q
```

**In plain English:** Use the physics model to extrapolate the state forward. The
uncertainty (P) grows because the model is not perfect (Q adds uncertainty).

### Update step

```
innovation     = z - H * x_predicted
S              = H * P * H^T + R
K              = P * H^T * S^{-1}         (Kalman gain)
x_updated      = x_predicted + K * innovation
P_updated      = (I - K * H) * P_predicted
```

**In plain English:** Compare the actual measurement to what the model predicted
(innovation). Compute how much to trust the measurement vs. the prediction (K). Blend
them. The uncertainty shrinks because we incorporated new information.

## In Rust

```rust
use machin_signal::kalman::{KalmanFilter, constant_velocity_1d};
use ndarray::{array, Array1};

// --- Quick start: track an object moving at constant velocity ---
let mut kf = constant_velocity_1d(
    0.1,   // process noise (how unpredictable is the motion?)
    1.0,   // measurement noise (how noisy is the GPS?)
    1.0,   // dt (seconds between updates)
);
// State = [position, velocity], observation = [position]

// Simulate noisy GPS readings of an object at position = 10 + 2*t
let measurements: Vec<Array1<f64>> = (0..20)
    .map(|t| {
        let true_pos = 10.0 + 2.0 * t as f64;
        array![true_pos + 0.5 * (t as f64 % 3.0 - 1.0)]  // add noise
    })
    .collect();

let states = kf.filter(&measurements);
let last = states.last().unwrap();
println!("Estimated position: {:.2}, velocity: {:.2}", last[0], last[1]);
// velocity converges to ~2.0

// --- Manual setup for custom state models ---
let mut kf = KalmanFilter::new(2, 1);  // state_dim=2, obs_dim=1
kf.transition = array![[1.0, 1.0], [0.0, 1.0]];   // F: constant velocity
kf.observation = array![[1.0, 0.0]];                // H: observe position only
kf.process_noise = array![[0.01, 0.0], [0.0, 0.01]]; // Q
kf.measurement_noise = array![[1.0]];                  // R

// Single predict-update cycle
kf.predict(None);                                     // no control input
kf.update(&array![5.0]);                              // measurement
println!("State: {:?}", kf.state);
println!("Covariance diagonal: {:?}", kf.covariance.diag());

// Or do both in one call
let estimated = kf.step(&array![5.1], None);
```

## When To Use This

| Technique | Best for | Limitations |
|-----------|----------|-------------|
| **Kalman filter** | Linear systems with Gaussian noise, sensor fusion | Assumes linearity and Gaussian noise |
| Extended Kalman Filter | Mildly nonlinear systems | Linearisation can diverge for strong nonlinearity |
| Particle filter | Highly nonlinear, non-Gaussian systems | Computationally expensive (O(N) particles) |
| Moving average | Simple smoothing with no physical model | No prediction capability; introduces lag |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `transition` (F) | How the state evolves between time steps | Derived from physics: constant velocity, constant acceleration, etc. |
| `observation` (H) | Which parts of the state can be measured | Set to 1 for directly observed states, 0 for hidden ones |
| `process_noise` (Q) | How much the model is wrong per step | Larger Q = less trust in the model, faster adaptation to changes |
| `measurement_noise` (R) | How noisy the sensor is | Larger R = less trust in measurements, smoother output |
| `state` | Initial state estimate | Best guess; the filter will correct it over a few steps |
| `covariance` (P) | Initial uncertainty | Start large (e.g., eye matrix) if you are unsure of the initial state |

## Pitfalls

1. **Q and R tuning.** The filter is only "optimal" when Q and R accurately reflect
   the true noise. If Q is too small the filter trusts the model too much and responds
   slowly to real changes. If R is too small it chases every noisy measurement.

2. **Linearity assumption.** The standard Kalman filter assumes F and H are constant
   matrices and noise is Gaussian. For nonlinear models, the filter can diverge.

3. **Observability.** If H does not provide enough information to reconstruct the full
   state, the covariance for unobserved states will not shrink. For example, measuring
   position alone cannot estimate acceleration without a model linking them.

4. **Numerical stability.** The covariance P must remain symmetric and positive-definite.
   Accumulated floating-point errors can break this. Periodically enforce symmetry:
   `P = (P + P^T) / 2`.

5. **Matrix inversion.** The update step inverts S. If S is singular (degenerate
   measurements), the filter will fail. The `constant_velocity_1d` helper avoids this
   for the common 1D tracking case.

## Going Further

- Combine multiple sensors (GPS + IMU + compass) by stacking their observations into a
  single H matrix and R block-diagonal.
- For time-varying models, update `kf.transition` at each step before calling `predict`.
- Use `machin_signal::filter::IirFilter::first_order_lowpass()` as a simpler alternative
  when you do not need a physical model -- just smoothing.
- Feed Kalman-filtered state estimates into `machin_chaos::lyapunov::lyapunov_spectrum`
  to detect chaotic dynamics in cleaned-up sensor data.
