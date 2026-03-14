# Markov Chains

## The Problem

You run a weather-dependent ice cream truck. Tomorrow's weather (Sunny, Cloudy, Rainy) depends only on today's weather -- not on last week's. If it is Sunny today, there is a 70% chance it is Sunny tomorrow and a 30% chance of Cloudy. You want to answer questions like:
- What is the probability of rain 5 days from now given that today is sunny?
- In the long run, what fraction of days are rainy?
- If it starts raining, how many days until the next sunny day on average?

Real-world scenarios:
- **Text generation:** Each word depends on the previous word. "I" is followed by "am" more often than "banana."
- **Customer behavior:** A user is either Active, Dormant, or Churned. Transition probabilities drive lifetime value models.
- **Queueing theory:** A server is Idle, Busy, or Overloaded. Predicting utilization requires knowing transition rates.
- **Financial modeling:** A credit rating migrates between AAA, AA, A, BBB, ... Default. Transition matrices price credit derivatives.
- **Board games:** The probability of landing on each square in Monopoly is a Markov chain over the 40 board positions.

## The Intuition

A Markov chain is a system that hops between states, where the next state depends *only on the current state* -- not on how you got there. This is the **Markov property** (memorylessness).

Think of it as a board game with loaded dice. You are on a square. You roll the dice, and based on the result, you move to the next square. The dice are *different* for each square (the transition probabilities), but they do not care about which squares you visited before.

The transition matrix P encodes all the dice. Row i is the probability distribution for "where do I go from state i?" Each row sums to 1.

Over many steps, something remarkable happens: no matter where you start, you end up visiting each state with a fixed frequency. That frequency is the **stationary distribution** -- the long-run equilibrium of the system.

## How It Works

### Transition Matrix

For N states, the transition matrix P is N x N:

```
P[i][j] = probability of going from state i to state j
```

Each row sums to 1 (row-stochastic). All entries are non-negative.

### State Distribution After k Steps

Starting from an initial distribution pi_0 (a probability vector over states):

```
pi_k = pi_0 * P^k
```

**In plain English:** Multiply the initial distribution by the transition matrix k times. Each multiplication advances one time step.

### Stationary Distribution

The stationary distribution pi* satisfies:

```
pi* = pi* * P
```

**In plain English:** If you are in the stationary distribution, one more step leaves you in the same distribution. It is the equilibrium. ix finds it via power iteration: start with a uniform distribution and keep multiplying by P until it stops changing.

### Mean First Passage Time

The expected number of steps to reach state j starting from state i:

```
M(i, j) = E[min{t > 0 : X_t = j | X_0 = i}]
```

**In plain English:** "On average, how many steps to get from i to j for the first time?" ix estimates this via Monte Carlo simulation (running many random walks and averaging).

### Ergodicity

A chain is **ergodic** if it is both irreducible (every state is reachable from every other state) and aperiodic (it does not cycle with a fixed period). Ergodic chains have a unique stationary distribution.

```
Ergodic = all entries of P^k are positive (for some k)
```

**In plain English:** After enough steps, there is a nonzero chance of being in *any* state, regardless of where you started.

## In Rust

```rust
use ix_graph::markov::MarkovChain;
use ndarray::{array, Array1};

// Weather model: Sunny=0, Cloudy=1, Rainy=2
let transition = array![
    [0.7, 0.2, 0.1],   // Sunny  -> 70% Sunny, 20% Cloudy, 10% Rainy
    [0.3, 0.4, 0.3],   // Cloudy -> 30% Sunny, 40% Cloudy, 30% Rainy
    [0.2, 0.3, 0.5],   // Rainy  -> 20% Sunny, 30% Cloudy, 50% Rainy
];

let mc = MarkovChain::new(transition)
    .unwrap()
    .with_names(vec![
        "Sunny".into(), "Cloudy".into(), "Rainy".into()
    ]);

// Q: If it is sunny today, what is the weather distribution in 5 days?
let today = array![1.0, 0.0, 0.0];  // 100% Sunny
let in_5_days = mc.state_distribution(&today, 5);
println!("After 5 days: {:.4}", in_5_days);
// e.g., [0.4861, 0.2714, 0.2425]

// Q: What is the long-run weather distribution?
let stationary = mc.stationary_distribution(1000, 1e-10);
println!("Stationary: {:.4}", stationary);
// Converges regardless of starting state

// Q: Simulate a 30-day weather sequence starting from Rainy
let path = mc.simulate(2, 30, 42);  // start=Rainy(2), 30 steps, seed=42
let names: Vec<&str> = path.iter()
    .map(|&s| mc.state_names[s].as_str())
    .collect();
println!("30-day forecast: {:?}", &names[..10]);  // first 10 days

// Q: On average, how many days from Rainy to Sunny?
let mfpt = mc.mean_first_passage(
    2,       // from: Rainy
    0,       // to: Sunny
    10_000,  // number of simulations
    1000,    // max steps per simulation
    42,      // seed
);
println!("Mean days Rainy -> Sunny: {:.1}", mfpt);

// Q: Is this chain ergodic?
let ergodic = mc.is_ergodic(100);
println!("Ergodic: {}", ergodic);  // true (all states reachable, aperiodic)
```

### Absorbing Chains

```rust
use ix_graph::markov::{MarkovChain, AbsorbingChain};
use ndarray::array;

// Customer lifecycle: Active=0, Dormant=1, Churned=2 (absorbing)
let transition = array![
    [0.7, 0.2, 0.1],   // Active  -> 70% stay active, 20% dormant, 10% churn
    [0.3, 0.4, 0.3],   // Dormant -> 30% reactivate, 40% stay dormant, 30% churn
    [0.0, 0.0, 1.0],   // Churned -> 100% stay churned (absorbing state)
];

let mc = MarkovChain::new(transition).unwrap();
let absorbing = AbsorbingChain::new(mc);

println!("Absorbing states: {:?}", absorbing.absorbing_states);  // [2]
println!("Is state 2 absorbing? {}", absorbing.is_absorbing_state(2));  // true
```

## When To Use This

| Model | Best When | Memory | State Space |
|---|---|---|---|
| **Markov chain** | Next state depends only on current state | Memoryless | Discrete, small-to-medium |
| **Hidden Markov model** | States are hidden, you observe emissions | Memoryless (hidden) | Discrete hidden + observed |
| **ARIMA** | Continuous time series with trends | Fixed-order | Continuous |
| **Recurrent neural network** | Long-range dependencies, large data | Learned (unlimited) | Continuous |

**Use Markov chains when:**
- The system has a clear set of discrete states.
- The memoryless assumption is reasonable (or a good approximation).
- You want analytical results (stationary distribution, mean passage times).
- The transition probabilities are known or can be estimated from data.

**Do not use when:**
- History matters (e.g., "the stock went up 3 days in a row" changes behavior).
- The state space is continuous or extremely large.
- You need to model hidden states (use an HMM instead -- see [hidden-markov-models.md](./hidden-markov-models.md)).

## Key Parameters

| Method | Parameters | Description |
|---|---|---|
| `MarkovChain::new(transition)` | `Array2<f64>` | Row-stochastic transition matrix. Returns `Result<Self, String>` |
| `.with_names(names)` | `Vec<String>` | Optional human-readable state labels |
| `.state_distribution(initial, steps)` | `Array1<f64>`, `usize` | Evolve a probability vector forward in time |
| `.stationary_distribution(max_iter, tol)` | `usize`, `f64` | Find equilibrium via power iteration. tol = convergence threshold |
| `.simulate(start, steps, seed)` | `usize`, `usize`, `u64` | Random walk returning `Vec<usize>` of visited states |
| `.mean_first_passage(from, to, n_sim, max_steps, seed)` | `usize`, `usize`, `usize`, `usize`, `u64` | Monte Carlo estimate of expected first passage time |
| `.is_ergodic(steps)` | `usize` | Check if P^steps has all positive entries |

## Pitfalls

1. **Rows must sum to 1.** `MarkovChain::new()` validates this with a tolerance of 1e-6. If you build the matrix from data (counting transitions), normalize each row: `row /= row.sum()`.

2. **Non-ergodic chains have no unique stationary distribution.** If the chain has absorbing states or periodic cycles, `stationary_distribution()` may converge to a distribution that depends on the starting point. Check `is_ergodic()` first.

3. **Mean first passage is estimated, not exact.** The `mean_first_passage()` method uses Monte Carlo simulation, so results vary with the seed and number of simulations. Use at least 10,000 simulations for stable estimates. For exact computation, solve the system of linear equations M = 1 + P * M (not yet implemented).

4. **Large state spaces.** The transition matrix is dense (N x N). A chain with 10,000 states uses 800 MB. For sparse chains, consider a sparse matrix representation (not yet supported).

5. **The Markov property is an assumption.** Real-world systems often have memory. A customer who has been dormant for 6 months is less likely to reactivate than one dormant for 1 week. Higher-order Markov chains (conditioning on the last k states) or HMMs can help.

6. **Periodic chains.** A chain that alternates deterministically (0 -> 1 -> 0 -> 1) has period 2. It has a stationary distribution ([0.5, 0.5]) but never actually converges to it -- it oscillates. `is_ergodic()` will return false.

## Going Further

- **Hidden Markov Models:** When you cannot observe the state directly. See [hidden-markov-models.md](./hidden-markov-models.md).
- **Viterbi algorithm:** Decode the most likely state sequence from observations. See [viterbi-algorithm.md](./viterbi-algorithm.md).
- **Continuous-time Markov chains (CTMC):** Transitions happen at random times (exponentially distributed) rather than at fixed steps. Model with rate matrices instead of transition matrices. Not yet in ix.
- **PageRank:** Google's original algorithm is a Markov chain on the web graph. The stationary distribution gives page importance.
- **Markov chain Monte Carlo (MCMC):** Design a Markov chain whose stationary distribution is the target distribution you want to sample from. Foundational for Bayesian inference.
- **Text generation:** Build a transition matrix from word-pair frequencies in a corpus. `simulate()` generates text.
