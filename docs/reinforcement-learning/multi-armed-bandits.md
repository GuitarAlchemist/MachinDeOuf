# Multi-Armed Bandits

## The Problem

You run an e-commerce site with three banner designs. Each visitor sees one banner and either clicks (reward = 1) or ignores it (reward = 0). Traditional A/B testing locks you into a fixed split -- say 33/33/33 -- for weeks before you have enough data to declare a winner. Meanwhile, every impression wasted on the worst banner is lost revenue.

Multi-armed bandits solve this: they *learn while earning*, gradually shifting traffic toward the best-performing variant without waiting for the experiment to "finish."

Other real-world scenarios:
- **Ad placement:** Which ad creative gets the most clicks?
- **Clinical trials:** Which drug dosage is safest while still gathering data on alternatives?
- **Content recommendation:** Which article headline drives the most engagement?
- **Dynamic pricing:** Which price point maximizes conversion?

## The Intuition

Imagine you walk into a casino with three slot machines. Each machine pays out at a different (unknown) rate. You have 1,000 tokens.

The naive strategy: play each machine 333 times, then pick the best one. But that wastes tokens on machines you already suspect are bad.

The smart strategy: keep playing the machine that *seems* best, but occasionally try the others in case your early impressions were wrong. That tension -- "stick with what works" vs. "maybe something better is out there" -- is the **exploration vs. exploitation trade-off**, and each bandit algorithm resolves it differently.

## How It Works

All three algorithms maintain an **estimated value** Q(a) for each arm a, updated incrementally after every pull.

### Incremental Mean Update

Every time arm a is pulled and returns reward r:

```
Q(a) <- Q(a) + (1/n) * (r - Q(a))
```

**In plain English:** Nudge the estimate toward the new observation. The nudge gets smaller as you collect more samples (1/n shrinks), so early pulls have a bigger effect than later ones.

### Epsilon-Greedy

```
With probability epsilon: pick a random arm        (explore)
With probability 1 - epsilon: pick argmax Q(a)     (exploit)
```

**In plain English:** Most of the time, go with the best-known option. But epsilon percent of the time, pick randomly to make sure you are not missing a better alternative.

### UCB1 (Upper Confidence Bound)

```
score(a) = Q(a) + sqrt(2 * ln(t) / N(a))
Pick: argmax score(a)
```

Where t is the total number of pulls so far and N(a) is how many times arm a has been pulled.

**In plain English:** Pick the arm with the highest *optimistic* estimate. The bonus term is large for arms you have not tried much (low N(a)), so under-explored arms get a boost. As you pull an arm more, the bonus shrinks and the raw average dominates.

### Thompson Sampling

```
For each arm a:
    sample theta(a) ~ Normal(mean=Q(a), variance=1/N(a))
Pick: argmax theta(a)
```

**In plain English:** Each arm has a belief distribution (how confident you are about its value). Sample one number from each distribution and pick the highest. Arms you know little about have wide distributions, so they occasionally produce high samples -- that is how Thompson explores. Arms you know well have tight distributions centered on their true value.

## In Rust

```rust
use machin_rl::bandit::{EpsilonGreedy, UCB1, ThompsonSampling};

// === Epsilon-Greedy: simple and predictable ===
let mut eg = EpsilonGreedy::new(
    3,    // 3 arms (banner designs A, B, C)
    0.1,  // explore 10% of the time
    42,   // RNG seed for reproducibility
);

for _ in 0..1000 {
    let arm = eg.select_arm();          // returns 0, 1, or 2
    let reward = simulate_click(arm);   // your reward signal
    eg.update(arm, reward);             // update Q-value for this arm
}
// Inspect learned values
println!("Estimated CTRs: {:?}", eg.q_values);   // Vec<f64>
println!("Pull counts:    {:?}", eg.counts);      // Vec<usize>


// === UCB1: no tuning parameter, automatic exploration ===
let mut ucb = UCB1::new(3);

for _ in 0..1000 {
    let arm = ucb.select_arm();         // deterministic given history
    let reward = simulate_click(arm);
    ucb.update(arm, reward);
}
println!("Total pulls: {}", ucb.total_count);


// === Thompson Sampling: Bayesian, often best in practice ===
let mut ts = ThompsonSampling::new(3, 42);

for _ in 0..1000 {
    let arm = ts.select_arm();
    let reward = simulate_click(arm);
    ts.update(arm, reward);
}
// Thompson tracks posterior means and variances
println!("Posterior means:     {:?}", ts.means);
println!("Posterior variances: {:?}", ts.variances);
```

See the full working example: [`examples/reinforcement-learning/bandits.rs`](../../examples/reinforcement-learning/bandits.rs)

## When To Use This

| Algorithm | Best When | Tuning Needed | Exploration Style |
|---|---|---|---|
| **Epsilon-Greedy** | You want simplicity; reward distributions are stable | Yes (epsilon) | Random uniform |
| **UCB1** | You want theoretical guarantees; no hyperparameters to tune | None | Optimism-driven |
| **Thompson Sampling** | You want best empirical performance; Bayesian framework is acceptable | None (seed only) | Posterior sampling |

**Use bandits instead of A/B tests when:**
- You cannot afford to waste traffic on bad variants during a long test.
- New variants are added or removed over time.
- You want to adapt to non-stationary reward distributions.

**Use A/B tests instead of bandits when:**
- You need strict statistical significance (p-values, confidence intervals).
- Regulatory requirements mandate a fixed protocol.

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_arms` | `usize` | -- | Number of options to choose from |
| `epsilon` (EpsilonGreedy) | `f64` | -- | Exploration rate: 0.0 = pure greedy, 1.0 = pure random. Typical: 0.05--0.15 |
| `seed` (EpsilonGreedy, Thompson) | `u64` | -- | RNG seed for reproducibility. UCB1 is deterministic so it needs no seed |

**Fields you can inspect after running:**

| Field | Type | Available On | Meaning |
|---|---|---|---|
| `q_values` | `Vec<f64>` | EpsilonGreedy, UCB1 | Running average reward per arm |
| `counts` | `Vec<usize>` | All three | How many times each arm was pulled |
| `total_count` | `usize` | UCB1 | Total pulls across all arms |
| `means` | `Vec<f64>` | ThompsonSampling | Posterior mean per arm |
| `variances` | `Vec<f64>` | ThompsonSampling | Posterior variance per arm (shrinks as you observe more) |

## Pitfalls

1. **Epsilon too high = too much exploration.** With epsilon = 0.5, half your traffic goes to random arms forever. Start around 0.1 and consider decaying it over time (not built-in; you would manually adjust `bandit.epsilon` between rounds).

2. **UCB1 explores aggressively at the start.** It must play every arm at least once before it starts using the confidence formula. With 100 arms, the first 100 pulls are purely exploratory.

3. **Thompson Sampling assumes Gaussian rewards.** This implementation uses a Normal posterior, which works well for continuous rewards (click-through rates, revenue). For binary rewards (click/no-click), a Beta-Bernoulli model would be more theoretically correct, but the Gaussian approximation works fine in practice with enough data.

4. **Non-stationary environments.** All three algorithms compute a running average, which weights early observations as heavily as recent ones. If the true reward rates change over time (e.g., seasonal effects), the algorithms adapt slowly. Consider a sliding window or exponential decay (modify the update step manually).

5. **Tied arms confuse greedy selection.** If two arms have identical Q-values, `select_arm()` always picks the one with the lower index. This is deterministic but may not be what you want.

## Going Further

- **Contextual bandits:** Choose arms based on user features (age, location). Not yet implemented, but you could combine bandit selection with a feature vector from `machin-supervised`.
- **Exploration vs. Exploitation deep dive:** See [exploration-vs-exploitation.md](./exploration-vs-exploitation.md) for a conceptual comparison of all three strategies.
- **Decaying epsilon:** Wrap the bandit in a loop that reduces `epsilon` over time: `bandit.epsilon = 1.0 / (round as f64).sqrt()`.
- **Regret analysis:** UCB1 has a provable O(sqrt(T * K * ln(T))) regret bound. Thompson Sampling often matches or beats it empirically.
- **Non-stationary bandits:** Use a fixed learning rate instead of 1/n: replace the update rule with `Q(a) <- Q(a) + alpha * (r - Q(a))` for a constant alpha (e.g., 0.1).
