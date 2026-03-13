//! Test Model Robustness
//!
//! Generate adversarial examples and evaluate defenses.
//!
//! ```bash
//! cargo run --example robustness_test
//! ```

use machin_adversarial::defense::{detect_adversarial, feature_squeezing};
use machin_adversarial::evasion::{fgsm, pgd};
use ndarray::array;

fn main() {
    let input = array![0.5, 0.3, 0.8, 0.1];
    let gradient = array![0.1, -0.2, 0.05, 0.3];

    // FGSM: single-step attack
    let adversarial = fgsm(&input, &gradient, 0.1);
    println!("Original:    {:?}", input);
    println!("FGSM attack: {:?}", adversarial);

    // PGD: iterative attack (stronger)
    let adversarial_pgd = pgd(&input, |_x| gradient.clone(), 0.1, 0.01, 40);
    println!("PGD attack:  {:?}", adversarial_pgd);

    // Defense: detect adversarial inputs via randomization
    let is_adversarial = detect_adversarial(
        &adversarial,
        |x| x * 2.0, // model function
        0.05,         // noise std
        100,          // samples
        0.5,          // threshold
        42,           // seed
    );
    println!("Detected as adversarial: {}", is_adversarial);

    // Defense: reduce precision to eliminate small perturbations
    let squeezed = feature_squeezing(&adversarial, 4); // 4-bit depth
    println!("Squeezed:    {:?}", squeezed);
}
