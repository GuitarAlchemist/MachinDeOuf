# Genetic Algorithms

## The Problem

You are designing an antenna for a satellite. The antenna shape is defined by 12 parameters (lengths, angles, curvatures). The signal quality for any given shape can be simulated, but the function mapping parameters to quality is wildly non-linear, full of local optima, and has no useful gradient. Calculus-based optimization (gradient descent) is useless here.

Real-world scenarios:
- **Circuit design:** Evolve component values (resistors, capacitors) to match a target frequency response.
- **Job-shop scheduling:** Evolve sequences of tasks across machines to minimize total completion time.
- **Neural architecture search:** Evolve network topologies (layer sizes, connections) to maximize accuracy.
- **Game character tuning:** Evolve NPC behavior parameters (aggression, speed, caution) for balanced gameplay.
- **Structural engineering:** Evolve truss configurations to minimize weight while meeting load requirements.

## The Intuition

Genetic algorithms are inspired by natural selection. Imagine you are breeding dogs for speed:

1. **Start with a random population** of dogs with varying leg lengths, muscle mass, and body shapes.
2. **Test fitness** by racing all dogs. The fastest ones are "fit."
3. **Select parents** -- fast dogs are more likely to breed (but slow dogs still have a small chance).
4. **Crossover** -- puppies inherit some traits from each parent. A puppy might get parent A's leg length and parent B's muscle mass.
5. **Mutation** -- occasionally, a random trait changes slightly. Maybe one puppy is a bit taller than either parent.
6. **Repeat** for many generations. Over time, the population converges on fast dogs.

The key insight: you never need to understand *why* certain traits make dogs fast. You just need to measure speed and let selection do the rest. This makes GAs ideal for problems where the objective function is a black box.

## How It Works

### The Algorithm

```
1. Initialize population of N random individuals
2. Evaluate fitness of each individual
3. Repeat for G generations:
   a. Select parents (tournament, roulette, or rank selection)
   b. Create offspring via crossover
   c. Apply mutation to offspring
   d. Evaluate fitness of offspring
   e. Replace population (keep elite individuals)
4. Return best individual found
```

### Selection

Three methods control which individuals become parents:

**Tournament selection:** Pick k random individuals, keep the best one.
```
parent = best of k randomly chosen individuals
```
**In plain English:** Run a mini-competition. Larger tournaments (higher k) put more pressure on selecting the best -- fewer weak individuals survive.

**Roulette selection:** Probability proportional to fitness.
```
P(select i) = (max_fitness - fitness_i) / sum(max_fitness - fitness_j)
```
**In plain English:** Spin a weighted wheel. Better individuals get a bigger slice. (Inverted for minimization -- lower fitness = bigger slice.)

**Rank selection:** Probability proportional to rank, not raw fitness.
```
P(select i) = rank(i) / sum(all ranks)
```
**In plain English:** The best gets rank N, the worst gets rank 1. This avoids the problem where one super-fit individual dominates roulette selection.

### Crossover (BLX-alpha)

For continuous optimization, ix uses Blend Crossover (BLX-alpha):

```
For each gene dimension d:
    lo = min(parent1[d], parent2[d]) - alpha * |parent1[d] - parent2[d]|
    hi = max(parent1[d], parent2[d]) + alpha * |parent1[d] - parent2[d]|
    child[d] = random uniform in [lo, hi]
```

With alpha = 0.5 (the default).

**In plain English:** The child's gene value is somewhere between the parents, plus a little wiggle room on either side. This lets offspring explore slightly beyond the range of their parents.

### Mutation (Gaussian)

```
For each gene with probability 0.3:
    gene += Normal(0, mutation_rate)
```

**In plain English:** Occasionally jiggle a gene value by a small random amount. The mutation rate controls how big the jiggle is.

### Fitness (Lower is Better)

ix uses minimization. The fitness function takes a candidate solution (a vector of f64) and returns a score. Lower scores are better.

```
fitness(x) = your_objective_function(x)  // e.g., sum of squared errors
```

## In Rust

```rust
use ix_evolution::genetic::GeneticAlgorithm;
use ndarray::Array1;

// Minimize the Sphere function: f(x) = sum(x_i^2)
// Global minimum is at the origin with f(0) = 0
let result = GeneticAlgorithm::new()
    .with_population_size(100)    // 100 candidate solutions per generation
    .with_generations(500)        // run for 500 generations
    .with_mutation_rate(0.15)     // Gaussian std dev for mutation
    .with_bounds(-5.0, 5.0)      // search space: each dimension in [-5, 5]
    .with_seed(42)                // reproducible results
    .minimize(
        &|x: &Array1<f64>| x.mapv(|v| v * v).sum(),  // fitness function
        3,  // 3 dimensions
    );

println!("Best solution: {:.4}", result.best_genes);
println!("Best fitness:  {:.6}", result.best_fitness);
println!("Generations:   {}", result.generations);
println!("Fitness curve: first={:.4}, last={:.4}",
    result.fitness_history.first().unwrap(),
    result.fitness_history.last().unwrap(),
);
```

### Harder Problem: Rastrigin Function

The Rastrigin function has many local optima, making it a standard test for global optimization:

```rust
use ix_evolution::genetic::GeneticAlgorithm;
use ndarray::Array1;
use std::f64::consts::PI;

let rastrigin = |x: &Array1<f64>| -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter()
        .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
        .sum::<f64>()
};

let result = GeneticAlgorithm::new()
    .with_population_size(200)
    .with_generations(1000)
    .with_mutation_rate(0.2)      // higher mutation to escape local optima
    .with_bounds(-5.12, 5.12)    // standard Rastrigin bounds
    .with_seed(123)
    .minimize(&rastrigin, 5);     // 5 dimensions

println!("Best fitness: {:.4} (global optimum is 0.0)", result.best_fitness);
```

### Custom Selection and the Individual Trait

The GA internally uses `RealIndividual` which implements the `Individual` trait:

```rust
use ix_evolution::traits::{Individual, RealIndividual};
use ix_evolution::selection;
use ndarray::array;
use rand::SeedableRng;

// Create individuals manually
let pop = vec![
    RealIndividual::new(array![1.0, 2.0]).with_fitness(10.0),
    RealIndividual::new(array![0.5, 0.5]).with_fitness(3.0),
    RealIndividual::new(array![0.1, 0.1]).with_fitness(0.5),  // best
];

let mut rng = rand::rngs::StdRng::seed_from_u64(42);

// Tournament selection: pick best of 3 random candidates
let parent = selection::tournament(&pop, 3, &mut rng);
println!("Selected fitness: {}", parent.fitness());

// Roulette selection: fitness-proportional
let parent = selection::roulette(&pop, &mut rng);

// Rank selection: rank-proportional
let parent = selection::rank(&pop, &mut rng);

// Crossover and mutation
let child = parent.crossover(&pop[0], &mut rng);
let mut mutated = child.clone();
mutated.mutate(0.1, &mut rng);
```

## When To Use This

| Method | Best When | Gradient Needed | Handles Local Optima |
|---|---|---|---|
| **Gradient descent** | Smooth, differentiable objective | Yes | No (gets stuck) |
| **Genetic algorithm** | Rugged landscape, black-box function | No | Yes (population diversity) |
| **Differential evolution** | Continuous optimization, fewer parameters to tune | No | Yes (often better than GA) |
| **Simulated annealing** | Single-solution search, simple to implement | No | Partially (random restarts help) |
| **Particle swarm** | Continuous, unimodal or mildly multimodal | No | Partially |

**Use GA when:**
- The fitness function is a black box (no gradient available).
- The search space has many local optima.
- You can afford many fitness evaluations (GAs are not sample-efficient).
- You want a population of diverse solutions, not just the single best.

**Do not use GA when:**
- The objective is smooth and differentiable (use gradient methods -- they are orders of magnitude faster).
- Fitness evaluation is extremely expensive (each generation evaluates the entire population).
- The problem dimension is very high (>100) -- GAs struggle in high dimensions.

## Key Parameters

| Parameter | Method | Default | Description |
|---|---|---|---|
| `population_size` | `with_population_size(n)` | 100 | Number of candidate solutions. Larger = more diversity but slower |
| `generations` | `with_generations(n)` | 500 | Number of evolutionary cycles. More = better convergence |
| `mutation_rate` | `with_mutation_rate(r)` | 0.1 | Std dev of Gaussian mutation. Higher = more exploration |
| `crossover_rate` | (internal) | 0.8 | Probability of crossover vs. cloning. Fixed at 0.8 |
| `tournament_size` | (internal) | 3 | Number of candidates in tournament selection |
| `elitism` | (internal) | 2 | Top-N individuals survive unchanged to next generation |
| `bounds` | `with_bounds(lo, hi)` | (-10, 10) | Search space bounds per dimension. Genes are clamped to this range |
| `seed` | `with_seed(s)` | 42 | RNG seed for reproducibility |

### EvolutionResult Fields

| Field | Type | Description |
|---|---|---|
| `best_genes` | `Array1<f64>` | The best solution vector found |
| `best_fitness` | `f64` | Fitness of the best solution (lower is better) |
| `generations` | `usize` | Number of generations run |
| `fitness_history` | `Vec<f64>` | Best fitness at each generation (should decrease monotonically due to elitism) |

## Pitfalls

1. **Premature convergence.** If the population becomes too homogeneous too quickly, the GA gets stuck in a local optimum. Fix: increase mutation rate, increase population size, or reduce tournament size (less selection pressure).

2. **Mutation rate too high = random search.** If mutation_rate is larger than the scale of the search space, offspring are essentially random. Keep it at 5-20% of the bound range.

3. **Elitism is critical.** Without elitism (keeping the best individuals unchanged), the best solution can be lost due to crossover and mutation. ix keeps the top 2 by default.

4. **Bounds must match the problem.** If the global optimum is at x = 100 but your bounds are [-10, 10], the GA will never find it. Always set bounds to cover the feasible region.

5. **Fitness evaluation dominates runtime.** The GA itself is fast. If your fitness function takes 1 second, each generation of 100 individuals takes 100 seconds. Consider parallelizing fitness evaluation (not built into ix's GA, but you can pre-evaluate and use `with_fitness()`).

6. **Not good for constrained optimization.** The GA has no built-in constraint handling. If your problem has constraints like "x1 + x2 <= 10," you must encode penalties into the fitness function.

## Going Further

- **Differential evolution:** A simpler evolutionary algorithm that often outperforms GAs on continuous optimization. See [differential-evolution.md](./differential-evolution.md).
- **Simulated annealing and PSO:** Available in `ix-optimize` for alternative global optimization strategies.
- **Custom Individual types:** Implement the `Individual` trait for non-continuous representations (permutations, bit strings, trees).
- **Adaptive mutation:** Decrease mutation rate over generations: `ga.with_mutation_rate(0.3 / (gen as f64 + 1.0).sqrt())`. Requires running the GA manually in a loop.
- **Island model:** Run multiple GA populations in parallel and occasionally migrate individuals between them for better diversity.
