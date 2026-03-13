# Evolutionary Dynamics

## The Problem

You are studying an ecosystem where hawks and doves compete for resources. Hawks fight aggressively; doves share peacefully. When two hawks meet, they fight and both get injured. When a hawk meets a dove, the hawk takes everything. When two doves meet, they split the resource.

Over many generations, what happens to the population mix? Will hawks take over? Will doves survive? Is there a stable balance?

Or: you are modeling a market where firms can choose aggressive pricing (hawk) or cooperative pricing (dove). Over time, firms that profit more grow; firms that lose shrink or exit. What is the long-run equilibrium?

These are questions about **evolutionary dynamics** -- how populations of strategies change over time when individuals interact, and successful strategies reproduce more.

## The Intuition

Imagine a large population where everyone plays a simple game against random opponents. Each individual uses a fixed strategy. After each round:

- Strategies that earned above-average payoffs **grow** (more individuals adopt them).
- Strategies that earned below-average payoffs **shrink**.

This is the **replicator dynamic** -- the evolutionary analogue of natural selection. It does not require biological evolution; any system where success breeds imitation follows the same math (memes spreading online, business strategies in markets, programming languages gaining adoption).

An **Evolutionarily Stable Strategy (ESS)** is a strategy that, once dominant in the population, cannot be invaded by a small group of mutants using a different strategy. Think of it as an equilibrium that is self-reinforcing.

The key difference from Nash equilibrium: Nash asks "can any *individual* improve by switching?" ESS asks "can a *small group of invaders* take over the population?" ESS is strictly stronger -- every ESS corresponds to a Nash equilibrium, but not every NE is an ESS.

## How It Works

### Replicator Dynamics

Let `x_i` be the fraction of the population using strategy `i`. The **payoff matrix** `A` defines the payoff of strategy `i` against strategy `j` as `A[i,j]`.

The fitness of strategy `i` in the current population:

```
f_i = sum_j A[i,j] * x_j
```

**In plain English:** your fitness is the average payoff you get when playing against a random member of the population.

The average fitness of the whole population:

```
f_avg = sum_i x_i * f_i
```

The replicator equation:

```
dx_i/dt = x_i * (f_i - f_avg)
```

**In plain English:** a strategy grows in proportion to how much better it does than average. If hawks earn more than the population average, the hawk fraction increases. If they earn less, it decreases. The `x_i` factor means that a strategy already at 0% stays at 0% (extinct strategies cannot spontaneously appear).

### Evolutionarily Stable Strategy (ESS)

Strategy `i` is an ESS if for every other strategy `j`:

**Condition 1 (Nash):** `A[i,i] >= A[j,i]`

The incumbent does at least as well against itself as any invader does against the incumbent.

**Condition 2 (Stability):** If `A[i,i] = A[j,i]`, then `A[i,j] > A[j,j]`

If the invader ties against the incumbent, the incumbent must do strictly better against the invader than the invader does against itself.

**In plain English:** (1) the incumbent strategy is a best response to itself, and (2) if there is a tie, the incumbent beats the invader in the mirror match.

### Hawk-Dove Game

The classic example with resource value `V` and fighting cost `C`:

|        | Hawk         | Dove    |
|--------|--------------|---------|
| **Hawk** | `(V-C)/2` | `V`     |
| **Dove** | `0`        | `V/2`   |

- If `V > C`: Hawk is ESS (fighting is worth it).
- If `V < C`: Neither pure strategy is ESS. The population converges to a mix of `V/C` hawks and `(1 - V/C)` doves.

## In Rust

The `machin-game` crate provides evolutionary dynamics using `ndarray`:

```rust
use machin_game::evolutionary::{
    replicator_dynamics, is_ess, find_ess,
    hawk_dove_matrix, rps_matrix,
    two_population_replicator,
};
use ndarray::{array, Array1, Array2};

fn main() {
    // --- Hawk-Dove Game ---
    let hd: Array2<f64> = hawk_dove_matrix(2.0, 4.0); // V=2, C=4 (V < C)

    // Check for ESS
    let ess_strategies: Vec<usize> = find_ess(&hd);
    println!("ESS strategies: {:?}", ess_strategies);
    // Empty -- neither pure Hawk nor pure Dove is ESS when V < C.

    println!("Is Hawk ESS? {}", is_ess(&hd, 0));  // false
    println!("Is Dove ESS? {}", is_ess(&hd, 1));  // false

    // Simulate replicator dynamics starting from 80% hawks, 20% doves.
    let initial = Array1::from_vec(vec![0.8, 0.2]);
    let trajectory: Vec<Array1<f64>> = replicator_dynamics(&hd, &initial, 0.01, 10_000);

    let final_state = trajectory.last().unwrap();
    println!("Final population: Hawk={:.2}%, Dove={:.2}%",
             final_state[0] * 100.0, final_state[1] * 100.0);
    // Converges to V/C = 50% Hawks, 50% Doves.

    // --- Prisoner's Dilemma ---
    // Defect dominates: defectors take over the population.
    let pd = array![[3.0, 0.0], [5.0, 1.0]]; // Cooperate, Defect
    let pd_initial = Array1::from_vec(vec![0.5, 0.5]);
    let pd_traj = replicator_dynamics(&pd, &pd_initial, 0.01, 5000);
    let pd_final = pd_traj.last().unwrap();
    println!("PD final: Cooperate={:.2}%, Defect={:.2}%",
             pd_final[0] * 100.0, pd_final[1] * 100.0);
    // Defect fraction -> ~100%

    // --- Rock-Paper-Scissors ---
    // Populations cycle without converging (no ESS exists).
    let rps: Array2<f64> = rps_matrix(1.0, -1.0, 0.0);
    let rps_initial = Array1::from_vec(vec![0.4, 0.3, 0.3]);
    let rps_traj = replicator_dynamics(&rps, &rps_initial, 0.01, 10_000);
    let rps_final = rps_traj.last().unwrap();
    println!("RPS final: R={:.2}, P={:.2}, S={:.2}",
             rps_final[0], rps_final[1], rps_final[2]);
    // All three strategies persist (cycle around 1/3 each).

    // --- Two-Population Replicator ---
    // Asymmetric game: predators vs. prey with different strategy sets.
    let pred_payoff = array![[3.0, 1.0], [2.0, 4.0]]; // Predator strategies vs prey
    let prey_payoff = array![[1.0, 4.0], [3.0, 2.0]]; // Prey strategies vs predator
    let pred_init = Array1::from_vec(vec![0.5, 0.5]);
    let prey_init = Array1::from_vec(vec![0.5, 0.5]);

    let (pred_traj, prey_traj) = two_population_replicator(
        &pred_payoff, &prey_payoff,
        &pred_init, &prey_init,
        0.01, 5000,
    );
    let pred_final = pred_traj.last().unwrap();
    let prey_final = prey_traj.last().unwrap();
    println!("Predators: {:?}", pred_final);
    println!("Prey: {:?}", prey_final);
}
```

### API summary

| Function | Signature | What it does |
|----------|-----------|--------------|
| `replicator_dynamics(payoff, initial, dt, steps)` | `-> Vec<Array1<f64>>` | Simulate population evolution, return full trajectory |
| `is_ess(payoff, strategy)` | `-> bool` | Check if pure strategy `i` is evolutionarily stable |
| `find_ess(payoff)` | `-> Vec<usize>` | Find all ESS among pure strategies |
| `hawk_dove_matrix(V, C)` | `-> Array2<f64>` | Classic Hawk-Dove payoff matrix |
| `rps_matrix(win, lose, draw)` | `-> Array2<f64>` | Rock-Paper-Scissors payoff matrix |
| `two_population_replicator(A, B, init_a, init_b, dt, steps)` | `-> (Vec<Array1>, Vec<Array1>)` | Asymmetric two-population dynamics |

### Reading the trajectory

The trajectory is a `Vec<Array1<f64>>` where each element is a snapshot of population proportions at one time step. The proportions always sum to 1.0 (they live on the probability simplex).

```rust
let traj = replicator_dynamics(&payoff, &initial, 0.01, 1000);

// Population at time step 0 (initial):
println!("{:?}", traj[0]);

// Population at time step 500:
println!("{:?}", traj[500]);

// Final population:
println!("{:?}", traj.last().unwrap());
```

## When To Use This

| Situation | Tool | Why |
|-----------|------|-----|
| Will strategy X dominate a population? | `replicator_dynamics` | Simulates natural selection dynamics |
| Is a strategy invasion-proof? | `is_ess` | Checks both Nash and stability conditions |
| Long-run equilibrium of a game | `replicator_dynamics` | Converges to stable fixed points |
| Asymmetric populations (predator-prey) | `two_population_replicator` | Separate dynamics for each population |
| One-shot strategic analysis | Nash equilibria instead | ESS is about populations, not individuals |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `payoff_matrix` | `Array2<f64>` | `A[i,j]` = payoff of strategy `i` against strategy `j`. Must be square for single-population dynamics. |
| `initial` | `Array1<f64>` | Starting population fractions. Must sum to 1.0. Must be > 0 for strategies you want to track (zero = extinct forever). |
| `dt` | `f64` | Time step for Euler integration. Smaller = more accurate, more steps needed. 0.01 is a good default. |
| `steps` | `usize` | Number of time steps. More = longer simulation. 5,000-10,000 is typical for convergence. |
| `V, C` (hawk-dove) | `f64` | Resource value and fighting cost. `V > C` = hawks win; `V < C` = mixed equilibrium at `V/C` hawks. |

### Choosing dt

The replicator equation is integrated with forward Euler. If `dt` is too large, the simulation can overshoot and produce negative populations (which are clamped to zero). Rules of thumb:

- `dt = 0.01` with `steps = 10,000` (simulates 100 time units) works for most games.
- If populations oscillate wildly or go to zero unexpectedly, reduce `dt` to 0.001.
- For games with large payoff values, reduce `dt` proportionally.

## Pitfalls

1. **Zero initial population stays zero.** The replicator equation has `dx/dt proportional to x`. If `x_i = 0` initially, that strategy can never appear. This is by design (no spontaneous mutation), but it means you must include all strategies you care about in the initial population, even at a tiny fraction like 0.01.

2. **ESS check is for pure strategies only.** `is_ess` checks whether a single pure strategy is evolutionarily stable. A *mixed* ESS (like the 50/50 hawk-dove equilibrium when V < C) is not detected by `find_ess`. Use `replicator_dynamics` to find the long-run population mix instead.

3. **Rock-Paper-Scissors cycles, does not converge.** The replicator dynamics for RPS orbit around the interior fixed point (1/3, 1/3, 1/3) without converging. In continuous time, orbits are neutrally stable. In discrete time (Euler), they can slowly spiral outward. Reduce `dt` if this becomes a problem.

4. **Payoff matrix must be consistent.** For single-population dynamics, the payoff matrix must be square (n strategies playing against n strategies). For two-population dynamics, the matrices can be rectangular.

5. **Replicator dynamics ignore mutation.** Unlike biological evolution, the replicator equation has no mutation term. Once a strategy goes extinct, it cannot return. If you need mutation, add a small uniform mixing term to the population update.

## Going Further

- **Hawk-Dove-Bourgeois:** Add a third strategy ("if I got here first, fight like a hawk; otherwise, retreat like a dove"). This is often the unique ESS.
- **Stochastic replicator dynamics:** Add noise to model finite populations where random drift matters. Important for small populations.
- **Nash equilibria:** The non-evolutionary theory of strategic interaction. See [Nash Equilibria](./nash-equilibria.md). Every ESS is a Nash equilibrium, but not conversely.
- **Mean-field games:** The `machin-game` crate also includes mean-field game theory for large populations with continuous strategy spaces. See `machin_game::mean_field`.
- Read: Hofbauer and Sigmund, *Evolutionary Games and Population Dynamics* (1998) -- the definitive textbook.
