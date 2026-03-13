# Particle Swarm Optimization

> A flock of birds searching for food. No bird knows where it is, but each one remembers the best spot it found and can see where the flock's best spot is. Together they converge on the richest field.

**Prerequisites:** [Probability & Statistics](../foundations/probability-and-statistics.md), [Vectors & Matrices](../foundations/vectors-and-matrices.md)

---

## The Problem

You are training a gradient boosting model to detect fraudulent transactions. The model has hyperparameters that cannot be learned by gradient descent: the number of trees, the maximum depth of each tree, the learning rate, and the regularization strength. You need to find the combination that minimizes cross-validation error on your held-out dataset.

Grid search is exhaustive -- if each of the 4 hyperparameters has 10 candidate values, that is 10,000 model trainings. Random search is better but directionless -- it does not learn from previous trials. You want an algorithm that explores the hyperparameter space intelligently, sharing information across trials so the search converges toward good regions.

Particle Swarm Optimization (PSO) does exactly this. It launches a swarm of candidate solutions ("particles"), each exploring the space independently but communicating their findings. The swarm self-organizes toward the optimum.

---

## The Intuition

Imagine 30 drones searching a mountain range for the lowest valley. Each drone:

1. **Remembers its personal best.** "The lowest altitude I have ever measured was at coordinates (x, y)."
2. **Knows the global best.** A radio channel broadcasts the lowest altitude any drone has found.
3. **Has momentum.** It does not teleport; it flies with velocity, carrying its previous direction.

At every time step, each drone adjusts its flight path based on three forces:

- **Inertia:** Keep going the way you were going. This prevents erratic zig-zagging.
- **Cognitive pull (personal best):** Steer toward the best location you personally found. "I found a good valley over there -- let me go back and explore nearby."
- **Social pull (global best):** Steer toward the best location anyone found. "The swarm found something even better -- let me head that direction."

The balance between these three forces determines the swarm's behavior:
- Heavy inertia and weak pulls: the swarm spreads out and explores broadly.
- Weak inertia and strong social pull: the swarm collapses onto the global best quickly (risks premature convergence).
- Balanced: the swarm explores efficiently and gradually converges.

Unlike gradient descent, PSO never computes a gradient. It only evaluates the objective function ("how good is this point?") and uses the swarm's collective memory to decide where to look next.

---

## How It Works

### Velocity Update

```
v_i(t+1) = w * v_i(t)
          + c1 * r1 * (pbest_i - x_i(t))
          + c2 * r2 * (gbest   - x_i(t))
```

**In plain English:** Each particle's new velocity is a weighted sum of three components:

- `w * v_i(t)`: **Inertia.** The particle keeps some of its current velocity. This is like a drone maintaining its heading. Higher `w` means the particle is harder to redirect.
- `c1 * r1 * (pbest_i - x_i(t))`: **Cognitive component.** The particle is pulled toward its own personal best position. `c1` controls the strength of this pull. `r1` is a random number between 0 and 1 that adds stochastic variation -- without it, all particles would follow deterministic paths and might all converge to the same local minimum.
- `c2 * r2 * (gbest - x_i(t))`: **Social component.** The particle is pulled toward the best position found by the entire swarm. `c2` controls this pull. `r2` is another random number.

### Position Update

```
x_i(t+1) = x_i(t) + v_i(t+1)
```

**In plain English:** Move the particle by adding the velocity. Then clamp it to the search bounds so particles cannot fly outside the allowed region.

### Personal and Global Best Update

After moving, each particle evaluates the objective at its new position. If the new position is better than its personal best, update `pbest_i`. If it is also better than the global best, update `gbest`.

**In plain English:** Every particle keeps a running record of the best spot it has visited. The swarm also keeps track of the single best spot any particle has ever found. These records are what drive the convergence -- they act as "beacons" pulling the swarm toward promising regions.

---

## In Rust

### Defining the Hyperparameter Search

```rust
use machin_optimize::pso::ParticleSwarm;
use machin_optimize::traits::{ClosureObjective, ObjectiveFunction};
use ndarray::Array1;

// Simulate cross-validation error as a function of 4 hyperparameters:
//   x[0] = learning rate     (optimal: 0.1)
//   x[1] = max depth         (optimal: 6.0)
//   x[2] = num trees         (optimal: 200.0)
//   x[3] = regularization    (optimal: 1.5)
//
// In production, each evaluation would train a model and compute CV error.
// Here we use a synthetic function with multiple local minima.
let cv_error = ClosureObjective {
    f: |x: &Array1<f64>| {
        // Shifted Ackley-like surface: global minimum at (0.1, 6.0, 200.0, 1.5)
        let targets = [0.1, 6.0, 200.0, 1.5];
        let shifted: Vec<f64> = x.iter().zip(&targets).map(|(xi, ti)| xi - ti).collect();
        let sum_sq: f64 = shifted.iter().map(|s| s * s).sum();
        let sum_cos: f64 = shifted.iter().map(|s| (2.0 * std::f64::consts::PI * s).cos()).sum();
        let n = shifted.len() as f64;
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    },
    dimensions: 4,
};
```

### Running PSO

```rust
let pso = ParticleSwarm::new()
    .with_particles(40)                // 40 candidate solutions exploring in parallel
    .with_max_iterations(500)          // 500 generations
    .with_bounds(-10.0, 300.0)         // search space (covers all hyperparameter ranges)
    .with_seed(42);                    // reproducible results

let result = pso.minimize(&cv_error);

println!("Best hyperparameters: {:?}", result.best_params);
println!("CV error: {:.6}", result.best_value);
println!("Iterations: {}", result.iterations);
println!("Converged: {}", result.converged);
// Expected: best_params close to [0.1, 6.0, 200.0, 1.5]
```

### Tuning the Swarm

The default parameters (`inertia=0.7`, `cognitive=1.5`, `social=1.5`) work well for most problems. For finer control, set them directly:

```rust
let mut pso = ParticleSwarm::new()
    .with_particles(60)
    .with_max_iterations(1000)
    .with_bounds(-10.0, 300.0)
    .with_seed(123);

// Access fields directly to tune behavior:
pso.inertia = 0.5;    // Less momentum -> faster convergence, less exploration
pso.cognitive = 2.0;   // Stronger personal memory -> more individual exploration
pso.social = 1.0;      // Weaker social pull -> less tendency to cluster prematurely
```

### Understanding the Return Value

`minimize` returns an `OptimizeResult`:

| Field         | Type          | Meaning                                          |
|---------------|---------------|--------------------------------------------------|
| `best_params` | `Array1<f64>` | The best parameter vector found by any particle   |
| `best_value`  | `f64`         | The objective value at `best_params`              |
| `iterations`  | `usize`       | How many generations were completed                |
| `converged`   | `bool`        | `true` if `best_value` dropped below `1e-12`      |

---

## When To Use This

| Situation | Use PSO? |
|-----------|----------|
| Hyperparameter tuning (no gradients available) | Yes -- PSO is designed for black-box optimization |
| Multi-modal landscape (many local minima) | Yes -- the swarm explores multiple regions simultaneously |
| Moderate dimensionality (2-50 parameters) | Yes -- PSO's sweet spot |
| You can evaluate the objective but not differentiate it | Yes -- PSO only calls `evaluate`, never `gradient` |
| Smooth, convex problem (linear regression) | No -- [gradient descent](gradient-descent.md) is faster and exact |
| Very high dimensionality (>100 parameters) | Cautious -- PSO needs exponentially more particles as dimensions grow |
| You need a single deterministic answer | No -- PSO is stochastic; use it for exploration, then refine with gradient methods |
| Budget is extremely tight (<100 evaluations) | No -- PSO needs enough iterations for the swarm to communicate and converge |

---

## Key Parameters

### Number of Particles (`with_particles`)

- More particles means better coverage of the search space but more evaluations per iteration.
- Rule of thumb: 20-50 particles for problems with fewer than 10 dimensions. 50-100 for higher dimensions.
- Default: `30`.

### Max Iterations (`with_max_iterations`)

- How many generations the swarm runs. Each generation evaluates every particle once.
- Total objective evaluations = `num_particles * max_iterations`. Budget accordingly.
- Default: `1000`.

### Bounds (`with_bounds(lo, hi)`)

- The search region for all dimensions. Particles are initialized uniformly within bounds and clamped after each move.
- Set bounds based on domain knowledge. Too tight and you might miss the optimum. Too wide and particles waste time in irrelevant regions.

### Inertia Weight (`inertia`, default `0.7`)

This is the most important PSO tuning parameter.

- **High inertia (0.9+):** Particles carry more momentum and explore widely. Good for the early phase of search.
- **Low inertia (0.4-0.5):** Particles respond quickly to personal and global bests. Good for convergence.
- **Adaptive strategy (not built-in):** Start high and linearly decrease to low over the course of the run. You can implement this by running PSO in stages with decreasing inertia.

### Cognitive Weight (`cognitive` / c1, default `1.5`)

- Controls how strongly each particle is pulled toward its own personal best.
- Higher c1 means more individual exploration. Each particle independently refines its own best region.
- If c1 is much larger than c2, particles act almost independently (less swarm behavior).

### Social Weight (`social` / c2, default `1.5`)

- Controls how strongly each particle is pulled toward the global best.
- Higher c2 means faster convergence -- the whole swarm rushes to the best known point.
- If c2 is much larger than c1, the swarm can prematurely converge to a local minimum because all particles cluster around one point.

### The c1/c2 Balance

| c1 vs c2 | Behavior |
|-----------|----------|
| c1 = c2 = 1.5 | Balanced exploration and convergence (recommended default) |
| c1 > c2 | More individual exploration, slower convergence, better for multi-modal problems |
| c1 < c2 | Fast convergence, risk of premature clustering, good for unimodal problems |
| c1 + c2 > 4.0 | Particles may oscillate wildly; reduce both or increase inertia |

### Seed (`with_seed`)

- PSO is stochastic (random initialization, random r1/r2 each step). Different seeds give different trajectories.
- For critical applications, run PSO with 5-10 different seeds and take the best result.

---

## Pitfalls

**Premature convergence.** The swarm collapses onto a local minimum because the social pull is too strong. All particles cluster together and stop exploring. Fix: increase `inertia`, increase `c1` relative to `c2`, or add more particles.

**Too many particles, too few iterations.** If you have 200 particles but only 50 iterations, each particle barely moves. The swarm never converges. PSO needs enough iterations for information to propagate. Fix: balance the budget (particles x iterations = total evaluations).

**Bounds too tight.** The optimum lies outside your specified bounds, so no particle can reach it. Always add margin to your bounds. If particles keep hitting the boundary (their best position is on the edge), widen the bounds.

**Bounds too wide.** With bounds from -1,000 to 1,000 and only 30 particles, the initial random positions are too spread out. Particles barely interact because they are far apart. Fix: tighten bounds based on domain knowledge, or increase particle count.

**Ignoring the cost of evaluation.** Each PSO iteration calls `evaluate` once per particle. If your objective function is expensive (e.g., training a full ML model), 40 particles x 500 iterations = 20,000 model trainings. Budget carefully. For expensive evaluations, consider Bayesian optimization or use PSO with very few particles (10-15) and more iterations.

**Using PSO for smooth, convex problems.** On a simple quadratic, gradient descent will find the exact minimum in a few hundred iterations. PSO will approximate it after thousands of evaluations. Use the right tool for the right problem.

---

## Going Further

- **See it in action:** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) minimizes the 10-dimensional Rosenbrock function using PSO -- a classic benchmark with a narrow curved valley.
- **Gradient-based alternative:** When gradients are available, [Gradient Descent (SGD, Momentum, Adam)](gradient-descent.md) will be faster and more precise.
- **Another gradient-free method:** [Simulated Annealing](simulated-annealing.md) uses a single agent with temperature-controlled randomness. It can be more efficient for lower-dimensional problems but lacks PSO's parallel exploration.
- **Evolutionary alternative:** [Genetic Algorithms](../evolutionary/genetic-algorithms.md) maintain a population and use crossover/mutation. They are more flexible for combinatorial (discrete) problems, while PSO naturally handles continuous spaces.
- **Hybrid approaches:** A powerful pattern is to run PSO first to find a good region, then refine with Adam. PSO provides the global exploration; Adam provides the local precision.
- **Distance metrics:** PSO implicitly uses Euclidean distance (particles move in Euclidean space). For non-Euclidean parameter spaces, see [Distance & Similarity](../foundations/distance-and-similarity.md).
