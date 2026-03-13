# Differential Evolution

## The Problem

You are calibrating a hydrological model that predicts river flow from rainfall. The model has 8 parameters (soil permeability, runoff coefficient, evaporation rate, etc.). You can run the simulation and compare predicted vs. observed flow, but the error surface is rugged -- small parameter changes sometimes cause large jumps in accuracy. Gradient descent fails because the function is not smooth. You need a robust, gradient-free optimizer that works reliably without much tuning.

Real-world scenarios:
- **Parameter calibration:** Fitting simulation parameters to observed data (physics models, financial models, biological models).
- **PID controller tuning:** Finding proportional, integral, and derivative gains for a control system.
- **Optical filter design:** Optimizing layer thicknesses in a multi-layer coating to achieve target transmittance.
- **Machine learning hyperparameter optimization:** Tuning learning rate, regularization, architecture choices.
- **Chemical process optimization:** Adjusting temperature, pressure, and flow rates to maximize yield.

## The Intuition

Differential evolution (DE) is like a group of hikers trying to find the lowest valley in a mountain range without a map.

Each hiker stands at a random point. To decide where to walk next, a hiker looks at three *other* hikers and computes a direction:

1. Pick three other hikers: A, B, and C.
2. Compute a direction: "Go from A toward where B is relative to C" -- that is, A + F * (B - C).
3. Try that new position. If it is lower than where you are now, move there. Otherwise, stay put.

This is remarkably effective because:
- The **step size adapts automatically.** When the population is spread out, (B - C) is large, so steps are large (exploration). As the population converges, (B - C) shrinks, so steps become fine-grained (exploitation).
- **No gradient needed.** Only fitness comparisons (is the new point better than the old one?).
- **Few parameters to tune.** Just F (mutation factor) and CR (crossover probability), and DE is robust to their exact values.

## How It Works

DE maintains a population of N vectors (candidate solutions) in D dimensions. The variant implemented in MachinDeOuf is **DE/rand/1/bin** (the most common).

### For Each Individual x_i in the Population

**Step 1: Mutation** -- Create a mutant vector:

```
v = x_r1 + F * (x_r2 - x_r3)
```

Where r1, r2, r3 are three distinct random indices (all different from i), and F is the mutation factor.

**In plain English:** Start at a random population member and take a step in the direction defined by the difference of two other members. F controls the step size.

**Step 2: Crossover** -- Create a trial vector by mixing the mutant with the original:

```
For each dimension j:
    if (random() < CR) or (j == j_rand):
        trial[j] = v[j]    (take from mutant)
    else:
        trial[j] = x_i[j]  (keep original)
```

Where CR is the crossover probability and j_rand ensures at least one dimension comes from the mutant.

**In plain English:** Randomly blend the original and mutant vectors. CR controls how much of the mutant makes it through. CR = 0.9 means 90% of dimensions come from the mutant.

**Step 3: Selection** -- Keep whichever is better:

```
if fitness(trial) <= fitness(x_i):
    x_i = trial
```

**In plain English:** Only move if the new position is at least as good. This greedy selection ensures the population never gets worse.

## In Rust

```rust
use machin_evolution::differential::DifferentialEvolution;
use ndarray::Array1;

// Minimize the Sphere function: f(x) = sum(x_i^2)
let result = DifferentialEvolution::new()
    .with_population_size(50)      // 50 candidate solutions
    .with_generations(1000)        // 1000 iterations
    .with_mutation_factor(0.8)     // F: step size scaling (0.5-1.0 typical)
    .with_crossover_prob(0.9)      // CR: how much of mutant to use
    .with_bounds(-5.0, 5.0)       // search range per dimension
    .with_seed(42)                 // reproducible
    .minimize(
        &|x: &Array1<f64>| x.mapv(|v| v * v).sum(),
        3,  // 3 dimensions
    );

println!("Best solution: {:.6}", result.best_genes);
println!("Best fitness:  {:.8}", result.best_fitness);
// DE typically finds fitness < 0.01 on the sphere function
```

### Harder Problem: Rosenbrock Function

The Rosenbrock function has a narrow, curved valley that challenges many optimizers:

```rust
use machin_evolution::differential::DifferentialEvolution;
use ndarray::Array1;

// Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
// Global minimum at (1, 1, ..., 1) with f = 0
let rosenbrock = |x: &Array1<f64>| -> f64 {
    (0..x.len() - 1)
        .map(|i| {
            100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2)
        })
        .sum::<f64>()
};

let result = DifferentialEvolution::new()
    .with_population_size(60)
    .with_generations(2000)
    .with_mutation_factor(0.8)
    .with_crossover_prob(0.9)
    .with_bounds(-5.0, 5.0)
    .with_seed(42)
    .minimize(&rosenbrock, 5);  // 5 dimensions

println!("Best fitness: {:.6} (optimal is 0.0)", result.best_fitness);
println!("Best genes: {:.4}", result.best_genes);
// Expected: all genes near 1.0
```

### Monitoring Convergence

```rust
// The fitness_history tracks the best fitness at each generation
let result = DifferentialEvolution::new()
    .with_population_size(30)
    .with_generations(500)
    .with_seed(42)
    .minimize(&|x: &Array1<f64>| x.mapv(|v| v * v).sum(), 3);

// Plot or analyze the convergence curve
let first_10: Vec<f64> = result.fitness_history[..10].to_vec();
let last_10: Vec<f64> = result.fitness_history[result.fitness_history.len()-10..].to_vec();
println!("First 10 gen fitness: {:?}", first_10);
println!("Last 10 gen fitness:  {:?}", last_10);
// Early: large improvements. Late: diminishing returns.
```

## When To Use This

| Method | Tuning Effort | Convergence Speed | Robustness | Best For |
|---|---|---|---|---|
| **Differential Evolution** | Low (F and CR) | Fast | Very high | Continuous parameter calibration |
| **Genetic Algorithm** | Medium (mutation, crossover, selection) | Moderate | High | Mixed-type or combinatorial problems |
| **Particle Swarm (PSO)** | Low | Fast on unimodal | Moderate | Smooth landscapes, low dimensions |
| **Simulated Annealing** | Medium (cooling schedule) | Slow | Moderate | Single-solution, any representation |
| **CMA-ES** | Very low | Very fast | Very high | Small-to-medium dimension continuous |

**Use DE when:**
- The problem is continuous optimization (real-valued parameters).
- You need a robust method that works out of the box.
- The fitness landscape is noisy, discontinuous, or multimodal.
- Dimension is moderate (2-100 parameters).

**Use GA instead when:**
- The search space is not continuous (permutations, discrete choices, tree structures).
- You want to use custom crossover operators.
- You need a diverse population for multi-objective optimization.

**Use gradient methods instead when:**
- The objective is smooth and differentiable -- gradient descent will be 100-1000x faster.

## Key Parameters

| Parameter | Method | Default | Typical Range | Description |
|---|---|---|---|---|
| `population_size` | `with_population_size(n)` | 50 | 5D to 10D | Number of candidate solutions. Rule of thumb: 5-10x the number of dimensions |
| `generations` | `with_generations(n)` | 1000 | 500-5000 | Number of iterations. More = better convergence |
| `mutation_factor` (F) | `with_mutation_factor(f)` | 0.8 | 0.4-1.0 | Scales the difference vector. Higher = larger steps, more exploration |
| `crossover_prob` (CR) | `with_crossover_prob(cr)` | 0.9 | 0.1-1.0 | Fraction of dimensions from mutant. Higher = more disruptive changes |
| `bounds` | `with_bounds(lo, hi)` | (-10, 10) | Problem-specific | Search range. Trial vectors are clamped to bounds |
| `seed` | `with_seed(s)` | 42 | Any u64 | RNG seed for reproducibility |

### Parameter Interaction Guide

| Scenario | Recommended F | Recommended CR |
|---|---|---|
| Separable function (dimensions independent) | 0.5-0.8 | 0.1-0.3 |
| Non-separable (dimensions coupled) | 0.8-1.0 | 0.9-1.0 |
| Noisy fitness | 0.4-0.6 | 0.5-0.7 |
| Many local optima | 0.8-1.0 | 0.9-1.0 |
| High dimension (D > 30) | 0.5-0.7 | 0.9-1.0 |

### EvolutionResult Fields

| Field | Type | Description |
|---|---|---|
| `best_genes` | `Array1<f64>` | The best solution vector found |
| `best_fitness` | `f64` | Fitness of the best solution (lower is better) |
| `generations` | `usize` | Number of generations run |
| `fitness_history` | `Vec<f64>` | Best fitness at each generation |

## Pitfalls

1. **Population too small = premature convergence.** With fewer individuals than dimensions, DE cannot generate enough diverse difference vectors. Use at least 5x the number of dimensions.

2. **F too low = stagnation.** With F < 0.4, the difference vectors are scaled down so much that the population barely moves. The algorithm converges too early on a suboptimal solution.

3. **F too high = instability.** With F > 1.2, steps overshoot, and trial vectors frequently land outside the useful region (they get clamped to bounds, losing information).

4. **CR too low on non-separable functions.** When CR is small, only 1-2 dimensions change per trial. If the function requires coordinated changes across dimensions (e.g., Rosenbrock's curved valley), low CR prevents the algorithm from moving diagonally in the search space.

5. **Same bounds for all dimensions.** If parameter 1 ranges from 0 to 1 and parameter 2 ranges from 0 to 10000, using `with_bounds(0.0, 10000.0)` wastes search effort on parameter 1. MachinDeOuf currently uses uniform bounds; for heterogeneous scales, normalize your parameters first.

6. **No early stopping.** The algorithm always runs for the full number of generations. If the fitness history plateaus early, you are wasting computation. Monitor `fitness_history` and add your own early stopping logic.

## Going Further

- **Genetic algorithms:** For combinatorial or mixed-type optimization, see [genetic-algorithms.md](./genetic-algorithms.md).
- **DE variants:** DE/best/1/bin uses the best individual instead of a random one for mutation (`v = x_best + F * (x_r1 - x_r2)`). Converges faster but may get trapped in local optima. Not yet implemented but easy to add.
- **Self-adaptive DE (jDE, SHADE):** Automatically adapt F and CR during the run. Eliminates the need to choose these parameters.
- **Constraint handling:** Penalize infeasible solutions in the fitness function: `fitness(x) + penalty * max(0, g(x))` where g(x) > 0 means constraint violated.
- **Multi-objective DE:** Maintain a Pareto front of non-dominated solutions. Combine with `machin-evolution`'s selection operators.
- **Hybrid methods:** Use DE for global exploration, then switch to a local optimizer (e.g., `machin-optimize`'s gradient methods) for fine-tuning around the best solution.
