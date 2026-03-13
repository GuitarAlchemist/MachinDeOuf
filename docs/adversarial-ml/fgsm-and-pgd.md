# Evasion Attacks: FGSM, PGD, C&W, and JSMA

## The Problem

You are testing a self-driving car's perception system before deployment. The camera-based
classifier identifies stop signs with 99.7% accuracy on the test set. But an adversary
could place a few stickers on a stop sign that are invisible to humans yet cause the
classifier to read "speed limit 45." Before you ship, you need to systematically probe
how much perturbation it takes to break the model -- and which inputs are most vulnerable.

Evasion attacks generate adversarial examples: inputs that look normal to humans but fool
machine learning models. They are essential for robustness testing in medical imaging,
malware detection, content moderation, and any safety-critical classification system.

## The Intuition

A neural network's decision boundary is a high-dimensional surface. Most of the time your
inputs sit comfortably far from this surface, and the model is confident. But the surface
has sharp ridges and narrow peninsulas that extend very close to normal inputs.

**FGSM** is a single, bold step toward the nearest ridge. Imagine shining a flashlight on
the decision boundary: the gradient tells you which direction it is, and you take one big
step in that direction. It is fast but crude.

**PGD** takes many small steps toward the boundary, projecting back onto an allowed
perturbation budget after each step. It is like walking toward a cliff but making sure you
never go more than epsilon metres from your starting point.

**C&W** is a careful optimiser that finds the *smallest* perturbation needed to cross the
boundary, balancing perturbation size against attack success. Think of it as a precision
tool vs PGD's sledgehammer.

**JSMA** works feature-by-feature, perturbing only the single most impactful pixel (or
feature) at each step. It produces sparse perturbations -- only a few pixels change.

## How It Works

### FGSM (Fast Gradient Sign Method)

```
x_adv = x + epsilon * sign(gradient_of_loss)
```

**In plain English:** Compute the gradient of the loss with respect to the input. Take
the sign of each component (+1 or -1). Step by epsilon in that direction. One shot, one
perturbation.

### PGD (Projected Gradient Descent)

```
for each step:
    x_adv = x_adv + alpha * sign(gradient)
    x_adv = project(x_adv, x, epsilon)   # clip to L-infinity ball
```

**In plain English:** Iterated FGSM with smaller steps (alpha < epsilon). After each
step, clip the perturbation so it stays within epsilon of the original input.

### C&W (Carlini & Wagner)

```
minimise:  ||delta||_2 + c * loss(x + delta, target)
```

**In plain English:** Find the smallest L2 perturbation that also minimises the
classification loss toward a target class. The trade-off constant c controls how hard
to push for misclassification vs keeping the perturbation small.

### JSMA (Jacobian-based Saliency Map)

```
for each step:
    compute saliency = impact of each feature on the target class
    perturb the most salient unmodified feature by theta
```

**In plain English:** Greedily pick the single feature that most increases the target
class probability. Perturb it. Repeat up to a budget of max_perturbations features.

## In Rust

```rust
use machin_adversarial::evasion::{fgsm, pgd, cw_attack, jsma};
use ndarray::array;

let input = array![0.5, 0.3, 0.8, 0.1];

// --- FGSM: single-step attack ---
let gradient = array![0.1, -0.2, 0.05, 0.3];
let adversarial = fgsm(&input, &gradient, 0.1);
// Each dimension moves by +/- 0.1 in the direction of the gradient sign
println!("FGSM: {:?}", adversarial);

// --- PGD: iterative attack (stronger) ---
let adversarial_pgd = pgd(
    &input,
    |_x| array![0.1, -0.2, 0.05, 0.3],  // gradient function
    0.1,   // epsilon (perturbation budget)
    0.01,  // alpha (step size per iteration)
    40,    // number of iterations
);
// Perturbation is guaranteed to stay within L-inf epsilon ball
let diff = &adversarial_pgd - &input;
for &d in diff.iter() {
    assert!(d.abs() <= 0.1 + 1e-10);
}

// --- C&W: minimum-perturbation attack ---
let adversarial_cw = cw_attack(
    &input,
    1,                           // target class
    |_x, _t| 1.0,               // loss function
    |x| x.clone(),              // gradient function
    1.0,                         // trade-off constant c
    50,                          // optimisation steps
    0.01,                        // learning rate
);

// --- JSMA: sparse perturbation ---
let adversarial_jsma = jsma(
    &input,
    1,                                      // target class
    |_x| array![0.1, 0.9, 0.3, 0.2],     // saliency function
    2,                                      // max features to perturb
    0.5,                                    // perturbation magnitude theta
);
// Only 2 features are modified
```

> Full runnable example: [examples/adversarial/robustness_test.rs](../../examples/adversarial/robustness_test.rs)

## When To Use This

| Attack | Speed | Perturbation quality | Best for |
|--------|-------|---------------------|----------|
| **FGSM** | Very fast (1 gradient) | Crude; often detectable | Quick sanity checks; adversarial training augmentation |
| **PGD** | Moderate (N gradients) | Strong L-inf bounded attack | Standard robustness benchmark; certification testing |
| **C&W** | Slow (optimisation loop) | Minimal L2 perturbation | Measuring true model vulnerability; security audits |
| **JSMA** | Moderate | Sparse (few features changed) | Feature importance analysis; interpretability |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `epsilon` | Maximum perturbation magnitude | 0.3 for MNIST (0-1 range); 8/255 for CIFAR; domain-specific |
| `alpha` (PGD) | Step size per iteration | alpha = epsilon / steps is a safe default |
| `steps` (PGD) | Number of iterations | 20-40 for most applications |
| `c` (C&W) | Trade-off: perturbation size vs attack success | Binary search over c is standard; start with 1.0 |
| `lr` (C&W) | Optimisation learning rate | 0.01 is typical; reduce if loss oscillates |
| `max_perturbations` (JSMA) | Budget: how many features to change | Lower = sparser; 10-20% of input dimensions |
| `theta` (JSMA) | How much to change each selected feature | Depends on feature scale |

## Pitfalls

1. **Gradient quality.** All these attacks require accurate gradients of the loss with
   respect to the input. If your model uses non-differentiable operations (argmax,
   discrete sampling), you need gradient approximations.

2. **Epsilon too large.** A perturbation visible to humans is not a meaningful adversarial
   example. Keep epsilon within the perceptual threshold for your domain.

3. **PGD local minima.** PGD can get stuck. Use random restarts (run PGD multiple times
   from different random perturbations of the input) for more reliable attacks.

4. **C&W is slow.** For large inputs (images), the optimisation loop is expensive. It is
   a gold-standard attack for evaluation, not for adversarial training data generation.

5. **JSMA assumes feature independence.** The greedy one-feature-at-a-time strategy may
   miss attacks that require coordinated changes across multiple features.

## Going Further

- Use `machin_adversarial::evasion::universal_perturbation` to find a single perturbation
  that fools the model on many inputs simultaneously.
- After generating adversarial examples, test defenses from
  `machin_adversarial::defense` -- see [adversarial-defenses.md](adversarial-defenses.md).
- Combine PGD with adversarial training: generate PGD examples at each training step and
  include them in the batch to harden the model.
- Feed adversarial examples into `machin_adversarial::poisoning::detect_label_flips`
  to check whether an attack on training data could go undetected.
