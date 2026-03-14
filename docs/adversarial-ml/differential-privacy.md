# Differential Privacy

## The Problem

You are building an analytics platform for a hospital network. Each hospital contributes
patient data to train a shared diagnostic model, but regulations forbid revealing
individual patient information. Even the model's output probabilities can leak private
data: an attacker who queries the model with slight variations of a patient's features
can infer whether that patient was in the training set (membership inference). You need
mathematical guarantees that no individual's data significantly affects the model's
outputs.

Differential privacy provides exactly this guarantee: the model behaves almost identically
whether any single patient is included or excluded from the training data.

## The Intuition

Imagine a survey where people answer "yes" or "no" to a sensitive question. Before
recording, each person privately flips a coin. If heads, they answer truthfully; if tails,
they flip again and answer "yes" for heads, "no" for tails. The aggregated results still
reflect the true distribution, but any individual's recorded answer is plausibly random.
This is the essence of differential privacy: add calibrated noise so that individual
contributions are hidden in the aggregate.

In ML, the noise is added to gradients during training (DP-SGD) or to model outputs at
inference time.

## How It Works

### Gaussian mechanism

```
noisy_gradient = gradient + N(0, sigma^2)
sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
```

**In plain English:** The sensitivity is the maximum amount any single training sample
can change the gradient. We add Gaussian noise scaled so that the probability of any
particular output changes by at most a factor of e^epsilon when one sample is added or
removed. Delta is the probability that the guarantee fails.

### Privacy budget (epsilon, delta)

- **epsilon:** Lower = more private. epsilon=1 is strong; epsilon=10 is weak.
- **delta:** Probability of a privacy breach. Typically 1/N^2 where N is dataset size.

### Additional protections

- **Temperature scaling:** Divide logits by a temperature > 1 before softmax. This
  produces flatter probability distributions that leak less information about individual
  training samples.
- **Prediction purification:** Zero out all but the top-K output classes, preventing
  an attacker from extracting information from low-probability classes.
- **Membership inference scoring:** Measure how confidently the model predicts a
  point's true label. High confidence suggests the point was in training.

## In Rust

```rust
use ix_adversarial::privacy::{
    differential_privacy_noise,
    model_confidence_masking,
    prediction_purification,
    membership_inference_score,
};
use ndarray::array;

// Add DP noise to a gradient
let gradient = array![1.0, 2.0, 3.0];
let noisy = differential_privacy_noise(
    &gradient,
    1.0,    // epsilon (privacy budget)
    1e-5,   // delta
    1.0,    // sensitivity (max gradient norm per sample)
    42,     // seed for reproducibility
);

// Temperature scaling to reduce information leakage
let logits = array![2.0, 1.0, 0.0];
let sharp = model_confidence_masking(&logits, 1.0);    // standard softmax
let smooth = model_confidence_masking(&logits, 10.0);  // flatter distribution

// Prediction purification: only expose top-1 class
let output = array![0.1, 0.7, 0.15, 0.05];
let purified = prediction_purification(&output, 1);
// Only the highest value (0.7) survives; rest become 0.0

// Membership inference risk assessment
let risk = membership_inference_score(
    |_x| array![0.1, 0.9, 0.0],  // model function
    &array![1.0, 2.0],            // test point
    1,                              // true label index
);
// risk = 0.9 (high confidence -> likely a training member)
```

## When To Use This

| Technique | Protects against | Trade-off |
|-----------|-----------------|-----------|
| **DP noise on gradients** | Membership inference, model inversion | Reduces model accuracy proportional to noise |
| **Temperature scaling** | Output-based information extraction | Reduces prediction sharpness |
| **Prediction purification** | Extracting class probabilities | Loses multi-class ranking information |
| **Membership inference scoring** | Auditing your own model's privacy risk | Diagnostic only; does not add protection |

## Pitfalls

1. **Privacy-accuracy trade-off.** Stronger privacy (smaller epsilon) requires more
   noise, which degrades model accuracy. There is no free lunch.
2. **Sensitivity estimation.** The Gaussian mechanism requires knowing the maximum
   gradient norm per sample. Underestimating sensitivity breaks the guarantee;
   overestimating adds unnecessary noise. Gradient clipping to a fixed norm is standard.
3. **Composition.** Each time you query the model or train for another epoch, you spend
   privacy budget. Track cumulative epsilon across all operations.
4. **Seed management.** Deterministic seeding is useful for reproducibility but must not
   be reused across different queries in production (it would produce identical noise).

## Going Further

- Implement DP-SGD by clipping per-sample gradients to a fixed norm, then adding
  `differential_privacy_noise` before the optimiser step. Use
  `ix_optimize` for the underlying SGD/Adam optimiser.
- Combine temperature scaling with prediction purification for layered defense.
- Use `membership_inference_score` as a pre-deployment audit: if many training samples
  have scores above a threshold, the model is leaking too much information.
- See [adversarial-defenses.md](adversarial-defenses.md) for complementary inference-time
  protections.
