# Hidden Markov Models

## The Problem

You are a doctor diagnosing a patient. The patient's true health state (Healthy, Developing Infection, Infected) is *hidden* -- you cannot observe it directly. What you *can* observe are symptoms: temperature, white blood cell count, energy level. Each health state produces symptoms with different probabilities. Given a sequence of daily symptom observations, you want to infer the most likely sequence of hidden health states.

Real-world scenarios:
- **Speech recognition:** Hidden states are phonemes, observations are acoustic features. The same phoneme sounds different depending on the speaker and context.
- **Gene finding:** Hidden states are coding/non-coding regions. Observations are nucleotide bases (A, T, C, G). Coding regions have different base frequencies.
- **Financial regime detection:** Hidden states are Bull/Bear/Sideways markets. Observations are daily returns. The same return could come from any regime.
- **Part-of-speech tagging:** Hidden states are grammatical tags (noun, verb, adjective). Observations are words. "Bank" can be a noun or a verb.
- **Activity recognition:** Hidden states are activities (walking, running, sitting). Observations are accelerometer readings.

## The Intuition

Imagine you are locked in a room and cannot see outside. A friend is in another room and can see the weather (Sunny or Rainy). Every hour, your friend does an activity -- Walk, Shop, or Clean -- and you can hear what they are doing. The weather influences their activity choice: on sunny days they walk more, on rainy days they clean more.

You observe a sequence: Walk, Shop, Clean, Walk. You want to figure out the weather sequence (Sunny/Rainy for each hour) that best explains these activities.

An HMM has three components:
1. **Initial distribution (pi):** What is the probability of each weather state at hour zero?
2. **Transition matrix (A):** Given the weather this hour, what is the probability of each weather next hour?
3. **Emission matrix (B):** Given the weather, what is the probability of each activity?

The "hidden" in HMM means you never see the weather directly -- only the activities it produces.

## How It Works

### The Three Problems of HMMs

**Problem 1: Evaluation** -- What is the probability of observing this sequence?

The **Forward algorithm** computes P(observations | model) efficiently by summing over all possible hidden state sequences:

```
alpha[t][i] = P(o_1, ..., o_t, q_t = i | model)
            = sum_j(alpha[t-1][j] * A[j][i]) * B[i][o_t]
```

**In plain English:** At each time step, the forward variable alpha[t][i] answers: "What is the joint probability of seeing the observations so far AND being in state i right now?" It computes this recursively: take the probability of being in each previous state, multiply by the transition probability to get here, then multiply by the probability of emitting the current observation.

P(observations) = sum of all alpha values at the final time step.

**Problem 2: Decoding** -- What is the most likely hidden state sequence?

The **Viterbi algorithm** finds the single best path through the hidden states (see [viterbi-algorithm.md](./viterbi-algorithm.md)).

The **Forward-Backward algorithm** computes the posterior probability of each state at each time step:

```
gamma[t][i] = P(q_t = i | observations, model)
            = alpha[t][i] * beta[t][i] / P(observations)
```

**In plain English:** gamma[t][i] is the probability that the system was in state i at time t, given *all* the observations (past and future). This is smoother than Viterbi -- instead of committing to one path, it gives a probability distribution over states at each time step.

**Problem 3: Learning** -- What parameters (A, B, pi) best explain the data?

The **Baum-Welch algorithm** (Expectation-Maximization for HMMs) iteratively re-estimates the parameters to increase the likelihood of the observations.

```
E-step: Compute gamma and xi using current parameters
M-step: Re-estimate A, B, pi from gamma and xi
Repeat until convergence
```

**In plain English:** Guess the parameters, compute what the hidden states probably were, then update the parameters to better match those guesses. Repeat. Each iteration is guaranteed to increase (or maintain) the likelihood.

## In Rust

### Creating an HMM

```rust
use machin_graph::hmm::HiddenMarkovModel;
use ndarray::{array, Array2};

// Weather HMM:
//   Hidden states: Rainy=0, Sunny=1
//   Observations:  Walk=0, Shop=1, Clean=2

let hmm = HiddenMarkovModel::new(
    array![0.6, 0.4],              // initial: 60% Rainy, 40% Sunny
    array![[0.7, 0.3],             // transition: Rainy->Rainy=0.7, Rainy->Sunny=0.3
           [0.4, 0.6]],            //            Sunny->Rainy=0.4, Sunny->Sunny=0.6
    array![[0.1, 0.4, 0.5],        // emission: Rainy->Walk=0.1, Shop=0.4, Clean=0.5
           [0.6, 0.3, 0.1]],       //           Sunny->Walk=0.6, Shop=0.3, Clean=0.1
).unwrap();

assert_eq!(hmm.n_states(), 2);
assert_eq!(hmm.n_observations(), 3);
```

### Problem 1: Evaluation (Forward Algorithm)

```rust
let observations = [0, 1, 2];  // Walk, Shop, Clean

// Log-probability of this observation sequence
let log_prob = hmm.forward(&observations);
println!("Log P(Walk,Shop,Clean) = {:.4}", log_prob);
// A finite negative number (log of a probability < 1)
```

### Problem 2a: Decoding (Viterbi)

```rust
let observations = [0, 1, 2, 0];  // Walk, Shop, Clean, Walk

let (path, log_prob) = hmm.viterbi(&observations);
println!("Most likely states: {:?}", path);
println!("Log probability:    {:.4}", log_prob);
// path[0] = 1 (Sunny) because P(Sunny)*P(Walk|Sunny) > P(Rainy)*P(Walk|Rainy)
// 0.4*0.6 = 0.24 > 0.6*0.1 = 0.06
```

### Problem 2b: Smoothing (Forward-Backward)

```rust
let observations = [0, 1, 2, 0, 1];

let gamma = hmm.forward_backward(&observations);
// gamma is a 5x2 matrix: gamma[t][i] = P(state i at time t | all observations)
println!("Posterior at t=0: Rainy={:.3}, Sunny={:.3}", gamma[[0, 0]], gamma[[0, 1]]);
println!("Posterior at t=2: Rainy={:.3}, Sunny={:.3}", gamma[[2, 0]], gamma[[2, 1]]);

// Each row sums to 1
for t in 0..5 {
    let sum: f64 = gamma.row(t).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

// MAP estimate: most probable state at each time step
let map_states = hmm.map_estimate(&observations);
println!("MAP states: {:?}", map_states);
```

### Problem 3: Learning (Baum-Welch)

```rust
// Observed sequence (longer sequences give better parameter estimates)
let observations = vec![0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0];

let log_prob_before = hmm.forward(&observations);

// Run Baum-Welch EM for up to 50 iterations, convergence tolerance 1e-8
let trained = hmm.baum_welch(&observations, 50, 1e-8).unwrap();

let log_prob_after = trained.forward(&observations);

println!("Log-likelihood before: {:.4}", log_prob_before);
println!("Log-likelihood after:  {:.4}", log_prob_after);
// After training, likelihood should increase (or stay the same)

println!("Learned transition:\n{:.4}", trained.transition);
println!("Learned emission:\n{:.4}", trained.emission);
```

See the full working example: [`examples/sequence/viterbi_hmm.rs`](../../examples/sequence/viterbi_hmm.rs)

## When To Use This

| Model | Hidden States | Observations | Transitions | Best For |
|---|---|---|---|---|
| **Markov chain** | No (states are observed) | N/A | State-to-state | Weather, pagerank, queueing |
| **HMM** | Yes (discrete) | Discrete | State-to-state | Speech, genes, POS tagging |
| **Kalman filter** | Yes (continuous) | Continuous | Linear + Gaussian | Tracking, navigation, finance |
| **CRF** | No (but discriminative) | Features | State-to-state | NLP tagging (superior to HMM) |
| **RNN/LSTM** | Learned | Any | Learned | Large data, complex dependencies |

**Use HMMs when:**
- You have a sequence of discrete observations.
- The underlying process has a small number of discrete hidden states.
- The Markov property is reasonable (state at t depends only on state at t-1).
- You want interpretable parameters (transition and emission matrices).

**Do not use when:**
- Observations are continuous and high-dimensional (use Kalman filters or RNNs).
- Long-range dependencies matter (HMMs are memoryless; LSTMs handle this).
- You have labeled training data (use CRFs or supervised models instead of EM).

## Key Parameters

### Constructor

| Parameter | Type | Constraints |
|---|---|---|
| `initial` | `Array1<f64>` | Length N, sums to 1, non-negative |
| `transition` | `Array2<f64>` | N x N, each row sums to 1, non-negative |
| `emission` | `Array2<f64>` | N x M, each row sums to 1, non-negative |

N = number of hidden states, M = number of observation symbols.

### Methods

| Method | Returns | Description |
|---|---|---|
| `forward(&[usize])` | `f64` | Log-probability of observation sequence |
| `forward_backward(&[usize])` | `Array2<f64>` | T x N posterior state probabilities (gamma) |
| `viterbi(&[usize])` | `(Vec<usize>, f64)` | Most likely state path + its log-probability |
| `map_estimate(&[usize])` | `Vec<usize>` | Most probable state at each time step (from gamma) |
| `baum_welch(&[usize], max_iter, tol)` | `Result<Self, String>` | EM-trained HMM with updated parameters |

### Baum-Welch Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `max_iter` | 50-200 | Maximum EM iterations |
| `tol` | 1e-6 to 1e-10 | Stop when log-likelihood change is below this |

## Pitfalls

1. **Observations must be `usize` indices.** The emission matrix maps states to observation *symbols* (integers 0..M-1). If your observations are continuous, you must discretize them first (e.g., bin temperatures into Low=0, Medium=1, High=2).

2. **Baum-Welch finds local optima.** EM is guaranteed to improve likelihood at each step but may converge to a local maximum. Run multiple times with different initializations and keep the best result.

3. **Label switching.** After Baum-Welch training, "state 0" and "state 1" may swap meaning compared to your initial model. The algorithm does not know which state is "Rainy" -- it just finds the best parameters. Inspect the emission matrix to interpret states.

4. **Underflow with long sequences.** The forward algorithm multiplies many small probabilities together. MachinDeOuf uses scaling (normalizing at each step) to prevent underflow, but extremely long sequences (>10,000 observations) may still lose precision.

5. **Zero probabilities block learning.** If `emission[i][k] = 0`, state i can never emit symbol k. Baum-Welch cannot recover from this. Initialize with small positive values everywhere (e.g., add 0.01 and re-normalize).

6. **Viterbi vs. MAP estimate.** Viterbi finds the single most probable *sequence* of states. MAP estimate finds the most probable state at each *individual* time step. These can differ! Viterbi ensures transition consistency; MAP does not.

## Going Further

- **Markov chains:** The observable foundation. See [markov-chains.md](./markov-chains.md).
- **Viterbi algorithm:** Deep dive into the dynamic programming approach. See [viterbi-algorithm.md](./viterbi-algorithm.md).
- **Gaussian HMMs:** Replace discrete emissions with continuous Gaussian distributions. Requires modifying the emission model (not yet in MachinDeOuf).
- **Higher-order HMMs:** The hidden state depends on the last k states, not just the last one. Increases model capacity at the cost of O(N^k) state space.
- **Input-Output HMMs:** Transition and emission probabilities depend on an external input signal. Useful for control applications.
- **Hierarchical HMMs:** States can themselves be HMMs, allowing multi-scale temporal modeling.
