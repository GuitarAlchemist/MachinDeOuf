# Viterbi Algorithm

## The Problem

You are building a GPS navigation system. The GPS chip gives you a noisy location reading every second, but the readings jump around (sometimes by 50 meters). You know the road network and the car's approximate speed. Given a sequence of noisy GPS readings, what is the most likely sequence of *actual* road segments the car traveled on?

This is a classic application of the Viterbi algorithm: you have a sequence of noisy observations and a model of how the hidden state evolves, and you want to find the single best hidden state sequence.

Real-world scenarios:
- **GPS path correction (map matching):** Snap noisy GPS points to the most likely road segments.
- **Speech recognition decoding:** Given acoustic features, find the most likely word sequence.
- **Error correction in communications:** Given a received (possibly corrupted) bit sequence, find the most likely transmitted sequence.
- **Gene annotation:** Given a DNA sequence, determine which regions are exons, introns, or intergenic.
- **Network intrusion detection:** Given a sequence of packet features, find the most likely sequence of system states (normal, probing, compromised).

## The Intuition

Imagine you are watching a friend walk through a foggy maze. You cannot see the maze walls, but you can hear footstep echoes. Some rooms produce loud echoes (large rooms), some produce quiet echoes (small rooms). You know the maze layout (which rooms connect to which) and the echo profile of each room.

Your friend takes 5 steps. You hear: loud, quiet, quiet, loud, quiet.

The brute-force approach: enumerate every possible 5-room path, compute its probability, and pick the best. With 10 rooms and 5 steps, that is 10^5 = 100,000 paths. With 100 rooms and 100 steps: 100^100. Impossible.

**Viterbi's insight:** Use dynamic programming. At each step, for each room, keep only the *single best path that ends in that room*. Discard all others. Why? Because if the best overall path passes through room X at step 3, the sub-path to room X at step 3 must also be the best way to reach room X at step 3. (This is the **principle of optimality.**)

This reduces the work from N^T (exponential) to N^2 * T (polynomial).

## How It Works

### Setup

Given an HMM with:
- N hidden states
- Initial distribution pi[i] = P(start in state i)
- Transition matrix A[i][j] = P(state j at t+1 | state i at t)
- Emission matrix B[i][k] = P(observation k | state i)
- Observation sequence O = [o_0, o_1, ..., o_{T-1}]

### The Algorithm

**Step 1: Initialization (t = 0)**

```
delta[0][i] = ln(pi[i]) + ln(B[i][o_0])
psi[0][i] = 0  (no predecessor)
```

**In plain English:** For each state, compute the log-probability of starting there and emitting the first observation. We work in log space to avoid multiplying many tiny probabilities (which causes underflow).

**Step 2: Recursion (t = 1, ..., T-1)**

```
delta[t][j] = max_i(delta[t-1][i] + ln(A[i][j])) + ln(B[j][o_t])
psi[t][j]   = argmax_i(delta[t-1][i] + ln(A[i][j]))
```

**In plain English:** For each state j at time t, find the predecessor state i that maximizes the path probability. Record the best predecessor in psi (the "backpointer"). delta[t][j] stores the log-probability of the best path ending in state j at time t.

**Step 3: Termination**

```
best_final_state = argmax_i(delta[T-1][i])
best_log_prob    = max_i(delta[T-1][i])
```

**In plain English:** The best path ends at whichever state has the highest delta at the last time step.

**Step 4: Backtracking**

```
path[T-1] = best_final_state
For t = T-2 down to 0:
    path[t] = psi[t+1][path[t+1]]
```

**In plain English:** Follow the backpointers from the end to the beginning to reconstruct the full path.

### Complexity

| Metric | Value |
|---|---|
| Time | O(N^2 * T) |
| Space | O(N * T) |

Where N = number of hidden states, T = length of observation sequence.

Compare this to brute force: O(N^T) -- exponential in sequence length.

## In Rust

### Basic Usage

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::array;

// States: Sunny=0, Rainy=1
// Observations: Walk=0, Shop=1, Clean=2
let hmm = HiddenMarkovModel::new(
    array![0.6, 0.4],                  // initial
    array![[0.7, 0.3], [0.4, 0.6]],    // transition
    array![[0.1, 0.4, 0.5],            // emission (Sunny)
           [0.6, 0.3, 0.1]],           // emission (Rainy)
).unwrap();

let observations = [0, 1, 2, 0];  // Walk, Shop, Clean, Walk

let (path, log_prob) = hmm.viterbi(&observations);
println!("Observations: Walk, Shop, Clean, Walk");
println!("Best states:  {:?}", path);       // e.g., [1, 0, 0, 1]
println!("Log prob:     {:.4}", log_prob);   // e.g., -5.2832
```

### Why Walk -> Sunny (Not Rainy)?

```rust
// At t=0, observation is Walk (index 0):
// P(Sunny) * P(Walk|Sunny) = 0.4 * 0.6 = 0.24
// P(Rainy) * P(Walk|Rainy) = 0.6 * 0.1 = 0.06
// Sunny wins despite lower prior, because Sunny emits Walk much more often.

let (path, _) = hmm.viterbi(&[0]);  // just "Walk"
assert_eq!(path, vec![1]);  // Sunny
```

### Deterministic HMM (Verifying Correctness)

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::array;

// Nearly deterministic mapping: state 0 -> obs 0, state 1 -> obs 1
let hmm = HiddenMarkovModel::new(
    array![1.0, 0.0],                    // always start in state 0
    array![[0.1, 0.9], [0.9, 0.1]],      // strongly alternating transitions
    array![[0.99, 0.01], [0.01, 0.99]],   // state i emits obs i with 99% prob
).unwrap();

let (path, log_prob) = hmm.viterbi(&[0, 1, 0, 1]);
assert_eq!(path, vec![0, 1, 0, 1]);  // perfectly decoded
println!("Log probability: {:.4}", log_prob);
```

### Viterbi vs. Forward-Backward (MAP)

These two decoding methods can give different results:

```rust
let observations = [0, 1, 2, 0, 1];

// Viterbi: best single path (globally consistent)
let (viterbi_path, _) = hmm.viterbi(&observations);

// MAP: most probable state at each individual time step
let map_path = hmm.map_estimate(&observations);

println!("Viterbi: {:?}", viterbi_path);
println!("MAP:     {:?}", map_path);
// These may differ! Viterbi ensures valid transitions between consecutive states.
// MAP picks the best state independently at each time step.
```

### GPS Map Matching (Conceptual)

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array1, Array2};

// Simplified GPS map matching:
// Hidden states: 4 road segments (0, 1, 2, 3)
// Observations: 3 GPS zones (0=North, 1=Central, 2=South)

// Which road segment you are on determines which GPS zone you likely see
let hmm = HiddenMarkovModel::new(
    array![0.5, 0.3, 0.15, 0.05],  // most likely starting on northern roads
    Array2::from_shape_vec((4, 4), vec![
        0.6, 0.3, 0.1, 0.0,  // road 0: likely stays or goes to road 1
        0.1, 0.5, 0.3, 0.1,  // road 1: central, connects to 0 and 2
        0.0, 0.2, 0.5, 0.3,  // road 2: connects south
        0.0, 0.0, 0.3, 0.7,  // road 3: southern, tends to stay
    ]).unwrap(),
    Array2::from_shape_vec((4, 3), vec![
        0.8, 0.15, 0.05,  // road 0 -> usually North GPS
        0.2, 0.6, 0.2,    // road 1 -> usually Central GPS
        0.05, 0.3, 0.65,  // road 2 -> usually South GPS
        0.02, 0.08, 0.9,  // road 3 -> usually South GPS
    ]).unwrap(),
).unwrap();

// Noisy GPS readings over 6 seconds
let gps_readings = [0, 0, 1, 1, 2, 2];  // North, North, Central, Central, South, South
let (road_segments, log_prob) = hmm.viterbi(&gps_readings);
println!("GPS zones:     {:?}", gps_readings);
println!("Road segments: {:?}", road_segments);
// Expected: [0, 0, 1, 1, 2, 2] or [0, 0, 1, 1, 2, 3] -- smooth path southward
```

See also: [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs)

## When To Use This

| Method | Finds | Complexity | Guarantees |
|---|---|---|---|
| **Viterbi** | Best single path | O(N^2 * T) | Globally optimal path |
| **Forward-Backward (MAP)** | Best state at each time step | O(N^2 * T) | Locally optimal per step |
| **Beam search** | Top-k paths | O(k * N * T) | Approximate (may miss optimal) |
| **Brute force** | Best path | O(N^T) | Optimal but intractable |

**Use Viterbi when:**
- You need the single most likely *complete* path (not just marginals at each step).
- Transition consistency matters (the path must follow valid transitions).
- The state space is small enough for exact computation (N < ~1,000).

**Use Forward-Backward instead when:**
- You want probability distributions over states at each time step.
- You want to quantify uncertainty (not just pick the best).
- You plan to use the posteriors for downstream tasks (e.g., Baum-Welch training).

## Key Parameters

Viterbi itself has no tuning parameters. It takes an HMM and an observation sequence and returns the optimal path. The quality of the result depends entirely on the HMM parameters.

| Input | Type | Description |
|---|---|---|
| `observations` | `&[usize]` | Sequence of observation indices (0..M-1) |

| Output | Type | Description |
|---|---|---|
| `path` | `Vec<usize>` | Most likely hidden state at each time step |
| `log_prob` | `f64` | Log-probability of the best path |

## Pitfalls

1. **Log-probability, not probability.** The returned `log_prob` is a large negative number (e.g., -15.3). The actual probability is exp(-15.3) which is tiny. Never exponentiate it for long sequences -- the result underflows to zero.

2. **Zero transitions block paths.** If `A[i][j] = 0`, the Viterbi algorithm will never transition from state i to j (log(0) = -infinity). Make sure your transition matrix allows all necessary transitions, even with small probabilities.

3. **Empty observation sequences.** `hmm.viterbi(&[])` returns `(vec![], 0.0)`. This is a valid edge case, not an error.

4. **Viterbi gives one path, but there may be many near-optimal paths.** If the top two paths have log-probabilities of -15.30 and -15.31, they are essentially equally likely. Viterbi only returns the first. For applications where near-ties matter, consider computing the N-best paths (not yet implemented).

5. **Viterbi and MAP can disagree.** Viterbi: "the best *sequence* of states." MAP: "the best *state* at each time step." Example: Viterbi picks state A at time t because it leads to a great path overall, even though state B is marginally more likely at time t in isolation. Both are correct answers to different questions.

6. **Numerical precision.** The implementation works in log space, which handles most underflow issues. However, with extremely small probabilities (e.g., `emission[i][k] = 1e-300`), even log space can lose precision. Keep probabilities above 1e-100 or so.

## Going Further

- **Hidden Markov Models:** Full treatment of HMMs, including Forward, Forward-Backward, and Baum-Welch. See [hidden-markov-models.md](./hidden-markov-models.md).
- **Markov chains:** The observable-state foundation. See [markov-chains.md](./markov-chains.md).
- **Conditional Random Fields (CRFs):** A discriminative alternative to HMMs that often performs better for sequence labeling when you have labeled training data.
- **Viterbi for convolutional codes:** In communications, the same algorithm decodes error-correcting codes. The "states" are encoder states and "observations" are received bits.
- **Lazy Viterbi:** For very large state spaces, expand only promising states. Related to A\* search -- see Q\* in `ix-search` for learned heuristic search.
- **Online Viterbi:** Process observations one at a time, emitting decoded states with a fixed delay. Useful for real-time applications like live speech recognition.
