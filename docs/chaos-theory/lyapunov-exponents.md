# Lyapunov Exponents

## The Problem

You manage a quantitative trading fund. Your models work well in calm markets but blow
up during turbulent periods. You need a numerical indicator that tells you, in real time,
whether the market regime is stable, periodic, or chaotic -- before losses accumulate.

Lyapunov exponents answer exactly this question for any dynamical system: "If I perturb
the current state by an infinitesimal amount, does the perturbation grow or shrink over
time?" They are the gold standard for detecting chaos in physics, climate science,
epidemiology, and engineering.

## The Intuition

Place two leaves next to each other on a river:

- **In a calm pool** (negative Lyapunov exponent), they drift together. Small differences
  shrink. The system is stable.
- **On a smooth current** (zero exponent), they stay the same distance apart. The system
  is periodic or quasi-periodic.
- **In rapids** (positive exponent), they fly apart exponentially fast. The system is
  chaotic -- tiny differences in starting position lead to wildly different outcomes.

The **maximal Lyapunov exponent (MLE)** measures the rate of this exponential divergence.
A positive MLE is the mathematical signature of chaos.

## How It Works

### MLE for a 1D map

For a discrete map x_{n+1} = f(x_n), the MLE is:

```
lambda = (1/N) * sum_{n=0}^{N-1} ln|f'(x_n)|
```

**In plain English:** At each step, compute the derivative (how much the function
stretches or compresses nearby points). Average the logarithm of the absolute derivative
along the entire orbit. If this average is positive, nearby orbits diverge exponentially
-- the system is chaotic.

### Lyapunov spectrum for continuous systems

For an n-dimensional ODE dx/dt = F(x), the full spectrum of n exponents is computed by:

1. Integrate the state x(t) forward using RK4.
2. Simultaneously evolve n perturbation vectors using the Jacobian matrix J(x).
3. Periodically re-orthogonalise the perturbation vectors (Gram-Schmidt) to prevent
   them from collapsing onto the most unstable direction.
4. The growth rate of each orthogonalised vector gives one Lyapunov exponent.

**In plain English:** Track how n independent tiny arrows evolve as they are carried
along by the flow. The fastest-growing arrow gives the MLE; the others fill out the
spectrum, telling you about the system's full geometry.

### Classification

| MLE value | `DynamicsType` | Meaning |
|-----------|---------------|---------|
| MLE < -threshold | `FixedPoint` | System converges to an equilibrium |
| -threshold <= MLE <= +threshold | `Periodic` | Limit cycle or quasi-periodic orbit |
| MLE > +threshold | `Chaotic` | Sensitive dependence on initial conditions |
| MLE > 10 | `Divergent` | System is blowing up (unstable) |

## In Rust

```rust
use ix_chaos::lyapunov::{mle_1d, lyapunov_spectrum, classify_dynamics, DynamicsType};

// --- 1D map: logistic map x_{n+1} = r*x*(1-x) ---
let r = 4.0;  // fully chaotic regime
let f  = |x: f64| r * x * (1.0 - x);
let df = |x: f64| r * (1.0 - 2.0 * x);  // derivative of f

let mle = mle_1d(f, df, 0.1, 10_000, 1000);
// mle ~ ln(2) ~ 0.693 for r=4.0
println!("MLE = {:.4}", mle);

let dynamics = classify_dynamics(mle, 0.01);
assert_eq!(dynamics, DynamicsType::Chaotic);

// --- Compare different r values ---
for &r in &[2.5, 3.2, 3.5, 3.9] {
    let f  = |x: f64| r * x * (1.0 - x);
    let df = |x: f64| r * (1.0 - 2.0 * x);
    let le = mle_1d(f, df, 0.1, 10_000, 1000);
    println!("r={:.1}: MLE={:.4} -> {:?}", r, le, classify_dynamics(le, 0.01));
}
// r=2.5: FixedPoint, r=3.2: FixedPoint (stable 2-cycle), r=3.5: Periodic, r=3.9: Chaotic

// --- Continuous system: Lyapunov spectrum for Lorenz ---
let lorenz_dynamics = |x: &[f64], _t: f64| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
};

let lorenz_jacobian = |x: &[f64], _t: f64| -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        -sigma,  sigma,  0.0,        // row 0
        rho - x[2], -1.0, -x[0],    // row 1
        x[1],    x[0],  -beta,       // row 2
    ]
};

let spectrum = lyapunov_spectrum(
    &lorenz_dynamics,
    &lorenz_jacobian,
    &[1.0, 1.0, 1.0],  // initial state
    0.01,                // dt
    10_000,              // steps
    1_000,               // transient to discard
);
println!("Lorenz spectrum: {:?}", spectrum);
// Expect: [~+0.9, ~0.0, ~-14.6] (one positive, one zero, one negative)
```

> Full runnable example: [examples/chaos/logistic_map.rs](../../examples/chaos/logistic_map.rs)

## When To Use This

| Technique | Best for | Limitations |
|-----------|----------|-------------|
| **MLE (1D map)** | Quick chaos detection in discrete maps | Requires the derivative f'(x) in closed form |
| **Lyapunov spectrum** | Full characterisation of continuous dynamical systems | Requires the Jacobian; expensive for high-dimensional systems |
| Bifurcation diagram | Visualising how dynamics change with a parameter | Qualitative; does not give a single number |
| Box-counting dimension | Measuring fractal structure of an attractor | Captures geometry, not dynamics |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `iterations` / `steps` | How long to average over | 10,000+ for reliable estimates; more for noisy systems |
| `transient` | Steps to discard before accumulating | 10--20% of total; ensures the orbit is on the attractor |
| `dt` | Integration step for continuous systems | Small enough for stable RK4 (0.01 for Lorenz) |
| `threshold` | Boundary between periodic and chaotic in `classify_dynamics` | 0.01 is a common default |
| `x0` / initial state | Starting point | Should be near the attractor; transient helps regardless |

## Pitfalls

1. **Transient matters.** If the orbit has not yet settled onto the attractor, the
   transient samples contaminate the exponent estimate. Always discard at least 1000
   initial iterations.

2. **Superstable orbits.** At certain parameter values (e.g., logistic map r=2), the
   derivative passes through zero. The MLE correctly returns negative infinity, but
   the logarithm of zero requires special handling (the implementation guards this).

3. **Finite-time vs asymptotic.** The true Lyapunov exponent is a limit as iterations go
   to infinity. Finite samples give an approximation. Always check convergence by
   increasing `iterations` and verifying the result stabilises.

4. **Jacobian accuracy.** For the spectrum, an incorrect Jacobian will produce wrong
   exponents. Double-check each partial derivative.

5. **High-dimensional systems.** The spectrum computation is O(n^2) per step (Gram-Schmidt
   on n vectors of dimension n). For n > ~20, consider only computing the MLE via the
   direct perturbation method.

## Going Further

- Plot MLE as a function of a parameter to build a **Lyapunov diagram** -- a quantitative
  companion to the bifurcation diagram.
- Use `ix_chaos::attractors::lorenz` to generate trajectories, then analyse their
  Lyapunov spectrum for different parameter values.
- Combine with `ix_chaos::fractal::correlation_dimension` to relate the number of
  positive exponents to the attractor's fractal dimension (Kaplan-Yorke conjecture).
- Feed financial returns into `mle_1d` with a suitable map to detect regime changes in
  real time.
