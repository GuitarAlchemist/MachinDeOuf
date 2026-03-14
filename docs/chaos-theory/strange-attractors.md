# Strange Attractors

## The Problem

You are simulating weather patterns for a climate research institute. Even with perfect
equations of motion, long-range forecasts diverge wildly from reality. Edward Lorenz
discovered this phenomenon in 1963: deterministic systems can produce trajectories that
never repeat, never diverge to infinity, and are exquisitely sensitive to initial
conditions. The geometric shapes these trajectories trace -- strange attractors -- are the
visual fingerprints of chaos.

Understanding attractors matters in turbulence modelling, cardiac rhythm analysis,
electronic circuit design, and any domain where deterministic unpredictability occurs.

## The Intuition

A **fixed point** is a ball sitting at the bottom of a bowl -- it settles and stays.
A **limit cycle** is a ball rolling around the rim of a bowl in a repeating loop.
A **strange attractor** is neither: the ball never settles, never repeats, but never
escapes either. It traces an infinitely complex pattern within a bounded region, like
a butterfly endlessly weaving through the same volume of air without ever retracing its
path.

The Lorenz attractor (the famous "butterfly") is the archetype. Its two lobes represent
two unstable equilibria that the trajectory visits in an unpredictable sequence -- like a
pinball bouncing between two bumpers forever.

## How It Works

### Lorenz system

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

**In plain English:** Three variables (temperature difference, fluid velocity, heat
transport) are coupled so that each one drives the others. With the classic parameters
(sigma=10, rho=28, beta=8/3), the system is chaotic: trajectories never repeat but remain
bounded on the butterfly-shaped attractor.

### Integration

All continuous-time attractors are integrated using the 4th-order Runge-Kutta method
(RK4):

```
k1 = f(state)
k2 = f(state + 0.5*dt*k1)
k3 = f(state + 0.5*dt*k2)
k4 = f(state + dt*k3)
new_state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

**In plain English:** Evaluate the derivative at four strategically chosen points, then
take a weighted average. This gives O(dt^4) accuracy per step -- far better than simple
Euler integration.

### Implemented attractors

| Attractor | Type | Dimension | Key behaviour |
|-----------|------|-----------|---------------|
| **Lorenz** | Continuous 3D | ~2.06 fractal | Two-lobe butterfly; sensitive to sigma, rho, beta |
| **Rossler** | Continuous 3D | ~2.01 fractal | Simpler single-band spiral; models chemical oscillations |
| **Chen** | Continuous 3D | Similar to Lorenz | Variant with different coupling structure |
| **Henon** | Discrete 2D | ~1.26 fractal | Classic 2D map; banana-shaped attractor |
| **Logistic map** | Discrete 1D | [0,1] interval | Period-doubling route to chaos |

## In Rust

```rust
use ix_chaos::attractors::{
    State3D, LorenzParams, RosslerParams, ChenParams, HenonParams,
    lorenz, rossler, chen, henon, logistic_map,
    integrate, rk4_step,
};

// --- Lorenz attractor ---
let params = LorenzParams::default(); // sigma=10, rho=28, beta=8/3
let initial = State3D::new(1.0, 1.0, 1.0);
let trajectory = lorenz(initial, &params, 0.01, 10_000);

// Trajectory stays bounded (strange attractor)
for s in &trajectory {
    assert!(s.x.abs() < 100.0 && s.y.abs() < 100.0 && s.z.abs() < 100.0);
}
println!("Lorenz: {} points, final = ({:.2}, {:.2}, {:.2})",
    trajectory.len(), trajectory.last().unwrap().x,
    trajectory.last().unwrap().y, trajectory.last().unwrap().z);

// --- Rossler attractor ---
let rossler_params = RosslerParams::default(); // a=0.2, b=0.2, c=5.7
let ross_traj = rossler(State3D::new(1.0, 1.0, 1.0), &rossler_params, 0.01, 10_000);

// --- Chen attractor ---
let chen_params = ChenParams::default(); // a=35, b=3, c=28
let chen_traj = chen(State3D::new(1.0, 1.0, 1.0), &chen_params, 0.001, 50_000);

// --- Henon map (discrete) ---
let henon_params = HenonParams::default(); // a=1.4, b=0.3
let henon_traj = henon(0.1, 0.1, &henon_params, 10_000);
for &(x, y) in &henon_traj[100..] {
    assert!(x.abs() < 3.0 && y.abs() < 3.0);  // bounded on attractor
}

// --- Logistic map ---
let orbit = logistic_map(0.5, 3.9, 1000);  // x0=0.5, r=3.9
let last = *orbit.last().unwrap();
assert!(last >= 0.0 && last <= 1.0);  // always in [0, 1]

// --- Custom ODE with integrate() and rk4_step() ---
let custom_deriv = |s: State3D| -> State3D {
    State3D::new(-s.y, s.x, -0.1 * s.z)  // simple harmonic + decay
};
let custom_traj = integrate(
    State3D::new(1.0, 0.0, 1.0),
    0.01,
    5000,
    &custom_deriv,
);

// Single RK4 step for manual control
let next = rk4_step(State3D::new(1.0, 0.0, 1.0), 0.01, &custom_deriv);
```

## When To Use This

| Attractor | Best for | Complexity |
|-----------|----------|------------|
| **Lorenz** | Canonical chaos example; weather/convection analogues | 3 ODEs, 3 parameters |
| **Rossler** | Simpler spiral chaos; chemical oscillation models | 3 ODEs, 3 parameters |
| **Chen** | Lorenz variant for comparative studies | 3 ODEs, 3 parameters |
| **Henon** | Discrete 2D chaos; fast to compute | 2D map, 2 parameters |
| **Logistic map** | Simplest chaos model; pedagogical use; 1D analysis | 1D map, 1 parameter |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `dt` | Integration time step | Lorenz: 0.01; Chen: 0.001 (stiffer). Too large = trajectory blows up |
| `steps` | Trajectory length | 10,000+ for a well-defined attractor shape |
| `sigma, rho, beta` | Lorenz: coupling strength, driving force, dissipation | Default (10, 28, 8/3) is the classic chaotic regime |
| `a, b, c` (Rossler) | Spiral tightness and folding | Default (0.2, 0.2, 5.7) gives the standard funnel |
| `a, b` (Henon) | Stretching and folding | Default (1.4, 0.3) gives the classic attractor |
| `r` (Logistic) | Bifurcation parameter | r < 3: fixed point; 3 < r < 3.57: period doubling; r > 3.57: chaos |

## Pitfalls

1. **dt too large.** RK4 is stable for moderate dt, but chaotic systems amplify errors.
   If the trajectory diverges to infinity, reduce dt by a factor of 10.

2. **Transient orbits.** The first few hundred points may be far from the attractor.
   Discard an initial transient (e.g., 1000 steps) before analysis or visualisation.

3. **Sensitivity to initial conditions.** Two trajectories starting 1e-10 apart will
   diverge exponentially on a chaotic attractor. This is a feature, not a bug -- but
   it means trajectory comparisons require careful alignment.

4. **Henon divergence.** The Henon map with non-default parameters (especially a > 1.4)
   can diverge to infinity. Always check boundedness.

5. **Logistic map domain.** The logistic map is only meaningful for x in [0, 1] and r in
   [0, 4]. Outside this range, orbits escape to negative infinity.

## Going Further

- Compute the Lyapunov exponent of each attractor with
  `ix_chaos::lyapunov::lyapunov_spectrum` to quantify how chaotic it is.
- Measure the fractal dimension of the attractor with
  `ix_chaos::fractal::box_counting_dimension_2d` or `correlation_dimension`.
- Use `ix_chaos::bifurcation::bifurcation_diagram` to visualise how the logistic
  map transitions from fixed point to period-doubling to chaos.
- Feed attractor trajectories into `ix_chaos::control::ogy_control` to stabilise
  unstable periodic orbits embedded in the chaotic attractor.
