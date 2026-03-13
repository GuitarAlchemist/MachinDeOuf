# Simulated Annealing

> Accept a worse answer now so you can find a better answer later -- then gradually stop accepting bad moves as you "cool down."

**Prerequisites:** [Calculus Intuition](../foundations/calculus-intuition.md), [Probability & Statistics](../foundations/probability-and-statistics.md)

---

## The Problem

You manage a warehouse with 50 workstations, 200 shelving zones, and 12 shipping docks. Every day, workers walk between zones to pick items for orders. The current layout evolved organically over a decade and is inefficient -- workers walk an average of 2.3 kilometers per shift just moving between zones. You want to rearrange the zones to minimize total walking distance.

This is a combinatorial optimization problem. There are more possible zone arrangements than atoms in the universe (200 factorial). You cannot try them all. Worse, the cost function (total daily walking distance) is not smooth -- moving one zone might improve things, moving two zones might make things worse, and the landscape is riddled with local minima (layouts that seem good but are far from optimal).

Gradient descent cannot help here: there are no gradients in a discrete layout problem. You need an algorithm that explores randomly, tolerates temporary setbacks, and gradually focuses in on good solutions.

---

## The Intuition

Simulated annealing is borrowed from metallurgy. When you heat metal and then cool it slowly, the atoms settle into a low-energy crystalline structure. If you cool it too fast, you get a brittle, disorganized mess.

The algorithm works the same way:

1. **Start hot.** At high temperature, the algorithm accepts almost any move -- even ones that make the solution worse. This lets it explore widely and escape bad neighborhoods.

2. **Evaluate neighbors.** At each step, make a small random change to the current solution (swap two zones, shift a workstation). Compute the new cost.

3. **Accept or reject.**
   - If the new solution is *better*, always accept it.
   - If the new solution is *worse*, accept it with a probability that depends on (a) how much worse it is and (b) the current temperature. High temperature means high acceptance probability. Low temperature means almost zero.

4. **Cool down.** Gradually reduce the temperature according to a schedule. Early on, the algorithm is an adventurous explorer. Late in the run, it is a cautious hill-climber that only accepts improvements.

The key insight is that accepting bad moves early on prevents you from getting stuck in the first local minimum you find. As the temperature drops, the algorithm naturally transitions from global exploration to local refinement.

---

## How It Works

### Acceptance Probability

```
If cost_new < cost_current:
    accept (always)

If cost_new >= cost_current:
    accept with probability P = exp(-delta / T)
    where delta = cost_new - cost_current
```

**In plain English:** When a neighbor is worse, compute how much worse (`delta`) and what the current temperature is (`T`). The probability of accepting the worse move is `e^(-delta/T)`. When `T` is large, this probability is close to 1 (accept almost anything). When `T` is tiny, this probability is close to 0 (reject almost everything worse). When `delta` is huge (much worse), the probability drops even at high temperature -- you still avoid catastrophically bad moves.

### Cooling Schedules

The cooling schedule determines how quickly the temperature drops. This is the main control knob beyond the initial temperature.

**Exponential cooling:** `T(k) = T0 * alpha^k`

**In plain English:** Multiply the temperature by a constant factor (e.g., 0.995) at each step. This is the most common schedule. It cools quickly at first, then slows down -- spending more time at low temperatures where the algorithm fine-tunes.

**Linear cooling:** `T(k) = T0 / (1 + alpha * k)`

**In plain English:** The temperature decreases by a fixed rate proportional to the step number. Simpler than exponential but can cool too fast in the early stages.

**Logarithmic cooling:** `T(k) = T0 / ln(1 + k)`

**In plain English:** The slowest schedule. Temperature drops very gradually. Theoretically guarantees finding the global optimum (given infinite time), but in practice you rarely have infinite time. Use this for problems where getting stuck in local minima is a serious concern and you can afford many iterations.

---

## In Rust

### Defining the Objective

Like all optimizers in MachinDeOuf, simulated annealing works with anything that implements `ObjectiveFunction`. For the warehouse layout problem, each parameter could represent a coordinate offset for a zone:

```rust
use machin_optimize::traits::{ClosureObjective, ObjectiveFunction};
use machin_optimize::annealing::{SimulatedAnnealing, CoolingSchedule};
use ndarray::{array, Array1};

// Simplified warehouse cost: total distance between zone pairs weighted by traffic.
// In reality, you'd compute Euclidean distances between zone positions
// multiplied by order frequency between each pair.
let warehouse_cost = ClosureObjective {
    f: |positions: &Array1<f64>| {
        // Rastrigin-like function: many local minima, one global minimum at origin.
        // This mimics a warehouse layout with many "pretty good" arrangements
        // but only one optimal one.
        let n = positions.len() as f64;
        10.0 * n
            + positions
                .iter()
                .map(|&x| x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos())
                .sum::<f64>()
    },
    dimensions: 6, // 3 zones, each with an (x, y) coordinate
};
```

### Running Simulated Annealing

```rust
let sa = SimulatedAnnealing::new()
    .with_temp(100.0, 1e-10)                              // start hot, cool to near-zero
    .with_cooling(CoolingSchedule::Exponential { alpha: 0.995 }) // multiply temp by 0.995 each step
    .with_max_iterations(50_000)                           // budget
    .with_step_size(0.5)                                   // radius of random perturbation
    .with_seed(42);                                        // reproducibility

let initial_layout = array![5.0, -3.0, 7.0, -1.0, 4.0, 2.0]; // random starting positions

let result = sa.minimize(&warehouse_cost, initial_layout);

println!("Best layout: {:?}", result.best_params);
println!("Cost: {:.4}", result.best_value);
println!("Iterations: {}", result.iterations);
println!("Converged (cooled fully): {}", result.converged);
```

### Trying Different Cooling Schedules

```rust
let schedules = vec![
    ("Exponential(0.995)", CoolingSchedule::Exponential { alpha: 0.995 }),
    ("Exponential(0.999)", CoolingSchedule::Exponential { alpha: 0.999 }),
    ("Linear(0.01)",       CoolingSchedule::Linear { alpha: 0.01 }),
    ("Logarithmic",        CoolingSchedule::Logarithmic),
];

let initial = array![5.0, -3.0, 7.0, -1.0, 4.0, 2.0];

for (name, schedule) in schedules {
    let sa = SimulatedAnnealing::new()
        .with_temp(100.0, 1e-10)
        .with_cooling(schedule)
        .with_max_iterations(50_000)
        .with_step_size(0.5)
        .with_seed(42);

    let result = sa.minimize(&warehouse_cost, initial.clone());
    println!("{:25} | cost: {:.4} | iters: {}", name, result.best_value, result.iterations);
}
```

### Understanding the Return Value

`minimize` returns an `OptimizeResult`:

| Field         | Type          | Meaning                                              |
|---------------|---------------|------------------------------------------------------|
| `best_params` | `Array1<f64>` | The best parameter vector found during the entire run |
| `best_value`  | `f64`         | The objective value at `best_params`                  |
| `iterations`  | `usize`       | How many steps were executed                          |
| `converged`   | `bool`        | `true` if temperature dropped below `min_temp`        |

Note that `best_params` tracks the global best, not just the current position. Even if the algorithm wanders away from a good solution (accepting a worse move), it remembers the best it ever found.

---

## When To Use This

| Situation | Use SA? |
|-----------|---------|
| Combinatorial problems (scheduling, layout, routing) | Yes -- SA excels here because it needs no gradients |
| Loss function with many local minima | Yes -- the random acceptance lets SA escape traps |
| You can compute the cost but not its gradient | Yes -- SA only calls `evaluate`, never `gradient` |
| Smooth, convex loss function (linear regression) | No -- [gradient descent](gradient-descent.md) will be faster and more precise |
| You need the provably optimal solution | Maybe -- SA gives good approximate solutions but no optimality guarantee in finite time |
| Very high-dimensional problems (>1000 parameters) | Cautious -- SA slows down because the neighborhood is vast; consider [Particle Swarm](particle-swarm.md) instead |

---

## Key Parameters

### Initial Temperature (`with_temp(initial, min)`)

- The initial temperature should be high enough that the algorithm accepts most moves early on. A common heuristic: set it so the acceptance probability for the "average bad move" is around 0.8.
- If you see the algorithm barely exploring (cost drops monotonically from the start), the temperature is too low.
- If the algorithm wanders randomly for most of the run and only improves near the end, the temperature is too high or the cooling is too slow.
- Typical range: `10.0` to `10_000.0` depending on the magnitude of your cost function.

### Minimum Temperature

- When `T < min_temp`, the algorithm stops and returns. Set this very low (`1e-8` to `1e-10`) unless you want early stopping.

### Cooling Schedule

- **Exponential with `alpha = 0.995`:** Good default. Finishes in about 1,000 steps (temp drops to ~0.7% of initial).
- **Exponential with `alpha = 0.999`:** Much slower cooling. Use with higher `max_iterations` for harder problems.
- **Linear:** Simpler. Useful when you know roughly how many iterations you can afford.
- **Logarithmic:** Theoretically optimal but extremely slow. Use when solution quality matters more than runtime.

### Step Size (`with_step_size`)

- Controls the standard deviation of the Gaussian perturbation added to each parameter. Larger values explore more aggressively.
- If the algorithm finds a good region but cannot refine it, reduce `step_size`.
- If the algorithm keeps producing neighbors with wildly different costs, reduce `step_size`.
- A good starting point: about 5-10% of the parameter range.

### Seed (`with_seed`)

- SA is stochastic. Different seeds give different runs. For important problems, run SA multiple times with different seeds and keep the best result.

---

## Pitfalls

**Cooling too fast.** The most common mistake. If `alpha` is too small (e.g., 0.9), the temperature crashes in a few dozen steps and the algorithm barely explores. It acts like greedy hill-climbing and gets stuck in the first local minimum it finds. Fix: increase `alpha` closer to 1.0 (e.g., 0.999) and increase `max_iterations`.

**Cooling too slow.** The algorithm wastes its entire budget wandering randomly because the temperature never drops enough to focus. Fix: decrease `alpha` (e.g., from 0.999 to 0.995) or increase `max_iterations`.

**Step size mismatch.** If `step_size` is much larger than the scale of your parameters, every neighbor is essentially random. If it is much smaller, the algorithm crawls. Match step size to the parameter range.

**Forgetting about scale.** The acceptance probability depends on `delta / T`. If your cost function values are in the millions, an initial temperature of 100.0 will accept almost nothing. Scale the temperature to match your cost function's magnitude.

**Running only once.** SA is stochastic. A single run might land in a mediocre local minimum. Best practice: run 5-10 independent runs with different seeds and take the best result.

**Not tracking the global best.** MachinDeOuf's implementation already does this -- `best_params` is the best ever seen, not the last visited. But if you implement your own SA loop, make sure to keep a separate `best_ever` variable.

---

## Going Further

- **Gradient-based alternative:** If your objective is smooth and differentiable, [Gradient Descent](gradient-descent.md) will converge faster and more precisely.
- **Population-based alternative:** [Particle Swarm](particle-swarm.md) explores with many agents simultaneously, which can be more efficient on multimodal landscapes.
- **Evolutionary approach:** [Genetic Algorithms](../evolutionary/genetic-algorithms.md) maintain a population and use crossover/mutation -- another way to balance exploration and exploitation.
- **Real-world example:** [`examples/optimization/pso_rosenbrock.rs`](../../examples/optimization/pso_rosenbrock.rs) demonstrates optimization on the Rosenbrock function; try replacing PSO with SA to compare.
- **Combinatorial extensions:** For discrete problems (traveling salesman, job scheduling), the perturbation step becomes a discrete swap instead of a Gaussian offset. The acceptance logic is identical.
