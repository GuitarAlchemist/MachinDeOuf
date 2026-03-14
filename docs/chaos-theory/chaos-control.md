# Chaos Control

## The Problem

A patient's heart is fibrillating -- the cardiac muscle contracts in a chaotic, ineffective
pattern instead of the regular rhythm needed to pump blood. Rather than shocking the heart
with a large defibrillation pulse, researchers have demonstrated that tiny, precisely timed
electrical nudges can stabilise one of the many unstable periodic orbits hidden within the
chaotic attractor, restoring normal rhythm.

Chaos control is the art of taming chaotic systems with minimal intervention. It applies
to stabilising lasers, controlling turbulent flows, managing chemical reactor oscillations,
and keeping communication systems locked to a carrier.

## The Intuition

A chaotic attractor is densely packed with unstable periodic orbits -- like a tangled ball
of yarn containing hidden loops of every length. Each loop is unstable: the system visits
it briefly before being flung away to a different part of the attractor.

**OGY method (Ott-Grebogi-Yorke):** Wait until the system naturally drifts close to the
desired unstable orbit, then apply a tiny parameter tweak that nudges it *onto* the orbit.
Because you are exploiting the system's own dynamics, the required perturbation is
vanishingly small. It is like balancing a ball on the tip of a hill by making microscopic
adjustments to the slope.

**Pyragas method (delayed feedback):** Add a continuous control signal proportional to
the difference between the current state and the state one period ago:
`control = K * (x(t - T) - x(t))`. When the system is on a period-T orbit, the control
signal is zero (no energy wasted). When it deviates, the feedback corrects it. This is
like a thermostat that compares the current temperature to what it was exactly one cycle
ago.

**Chaos synchronisation:** Two identical chaotic systems, started from different initial
conditions, will follow completely different trajectories. But by coupling one variable
of the "response" system to the "driver" system, you can force them to synchronise. This
is the basis for chaos-based secure communication.

## How It Works

### OGY control (discrete maps)

For a map x_{n+1} = f(x_n, r) with an unstable fixed point x* at parameter r_0:

```
delta_r = -(df/dx) / (df/dr) * (x_n - x*)
r_n = r_0 + clamp(delta_r, -max_perturbation, +max_perturbation)
```

**In plain English:** When the orbit passes near the target fixed point, shift the
parameter by an amount proportional to the deviation. The ratio of partial derivatives
determines the optimal direction. The perturbation is clamped to prevent large kicks.

### Pyragas delayed feedback (continuous systems)

```
dx/dt = F(x) + K * (x(t - tau) - x(t))
```

**In plain English:** The control term is zero when x(t) = x(t - tau), i.e., when the
system is on a period-tau orbit. Any deviation from periodicity generates a restoring
force proportional to K.

### Drive-response synchronisation

```
dx_response/dt = F(x_response) + coupling * (x_driver - x_response)
```

**In plain English:** The response system follows its own chaotic dynamics, plus a
correction that pulls it toward the driver whenever they diverge. With sufficient
coupling strength, the response locks onto the driver's trajectory.

## In Rust

```rust
use ix_chaos::control::{
    ogy_control, pyragas_control, drive_response_sync,
};

// --- OGY: stabilise the logistic map ---
let r = 3.8;  // chaotic regime
let target = 1.0 - 1.0 / r;           // unstable fixed point
let df_dx = r * (1.0 - 2.0 * target); // partial df/dx at (x*, r)
let df_dr = target * (1.0 - target);   // partial df/dr at (x*, r)

let trajectory = ogy_control(
    |x, r| r * x * (1.0 - x),  // the logistic map
    target,                      // desired fixed point
    r,                           // nominal parameter
    df_dx, df_dr,                // linearisation at target
    0.1,                         // max parameter perturbation
    0.5,                         // initial state
    200,                         // total steps
    50,                          // start control at step 50
);

// After control engages, x converges to target
let (last_x, last_r) = trajectory.last().unwrap();
println!("Stabilised at x={:.4} (target={:.4}), r={:.4}", last_x, target, last_r);

// --- Pyragas: stabilise a continuous system ---
let oscillator = |x: &[f64]| -> Vec<f64> {
    vec![x[1], -x[0] + 0.3 * x[1] * (1.0 - x[0] * x[0])]  // van der Pol
};

let controlled = pyragas_control(
    &oscillator,
    &[1.0, 0.0],   // initial state
    0.01,           // dt
    5000,           // steps
    628,            // delay ~ one period (2*pi / 0.01)
    0.5,            // feedback gain K
    1000,           // start control after 1000 steps
);
println!("Pyragas: {} steps, final state = {:?}",
    controlled.len(), controlled.last().unwrap());

// --- Synchronise two Lorenz systems ---
let lorenz = |x: &[f64]| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
};

let (driver, response, errors) = drive_response_sync(
    &lorenz,
    &[1.0, 1.0, 1.0],    // driver initial
    &[5.0, 5.0, 5.0],    // response initial (different!)
    0.01,                  // dt
    5000,                  // steps
    5.0,                   // coupling strength
    &[0],                  // couple x-variable only
);
println!("Initial sync error: {:.4}", errors[0]);
println!("Final sync error:   {:.4}", errors.last().unwrap());
```

## When To Use This

| Method | Best for | Requirements |
|--------|----------|-------------|
| **OGY** | Discrete maps; stabilising fixed points | Know the map, its derivative, and the target fixed point |
| **Pyragas** | Continuous systems; stabilising periodic orbits | Know the approximate period; no need for explicit equations |
| **Drive-response** | Synchronising two identical chaotic systems | Identical dynamics; sufficient coupling strength |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `max_perturbation` (OGY) | Limits the parameter adjustment per step | Smaller = less invasive but slower convergence |
| `delay_steps` (Pyragas) | Must match the period of the target orbit | Estimate period from the uncontrolled trajectory |
| `gain` (Pyragas) | Feedback strength | Too small = no control; too large = overshoot and instability |
| `coupling_strength` (sync) | How strongly the response is pulled toward the driver | Must exceed a system-dependent threshold for synchronisation |
| `control_start` | When control begins | Allow transient dynamics to settle before engaging control |

## Pitfalls

1. **OGY only works near the target.** The method relies on linearisation, which is only
   valid in a small neighbourhood of the fixed point. The system must naturally drift close
   before control can engage. This can take many iterations.

2. **Wrong period estimate breaks Pyragas.** If the delay does not match the actual orbit
   period, the feedback fights the natural dynamics instead of reinforcing the orbit.

3. **Coupling threshold.** Below a critical coupling strength, drive-response
   synchronisation fails completely. The threshold depends on the system's Lyapunov
   exponents.

4. **Euler integration.** The Pyragas implementation uses Euler integration for
   simplicity. For stiff systems, this may require very small dt. Consider wrapping
   the dynamics in an RK4 integrator from `ix_chaos::attractors::rk4_step` for
   better accuracy.

## Going Further

- Combine OGY with `ix_chaos::lyapunov::mle_1d` to verify that the controlled orbit
  has a negative Lyapunov exponent (confirming stabilisation).
- Use `ix_chaos::bifurcation::bifurcation_diagram` to identify all available unstable
  periodic orbits before choosing a target for OGY control.
- Explore chaos-based communication: encode information in the driver system's parameters,
  then decode by measuring synchronisation error in the response.
- Combine Pyragas control with `ix_signal::kalman::KalmanFilter` to estimate the
  system state from noisy observations before applying the feedback.
