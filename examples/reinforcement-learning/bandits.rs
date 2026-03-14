//! A/B Testing with Multi-Armed Bandits
//!
//! Decide which variant wins using Thompson sampling.
//!
//! ```bash
//! cargo run --example bandits
//! ```

use ix_rl::bandit::ThompsonSampling;

fn main() {
    // 3 variants: A, B, C with true click-through rates
    let true_rates = [0.05, 0.12, 0.08]; // B is actually best
    let mut bandit = ThompsonSampling::new(3, 42);

    // Simulate 1000 trials
    for _ in 0..1000 {
        let arm = bandit.select_arm();
        // Simulate a Bernoulli reward based on true rate
        let reward = if rand::random::<f64>() < true_rates[arm] {
            1.0
        } else {
            0.0
        };
        bandit.update(arm, reward);
    }

    println!("Best variant: {} (expected: 1)", bandit.best_arm());
    println!("Estimated values: {:?}", bandit.estimated_values());
    println!("True rates: {:?}", true_rates);
}
