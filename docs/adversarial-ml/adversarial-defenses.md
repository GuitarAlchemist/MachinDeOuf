# Adversarial Defenses

## The Problem

You are deploying a medical imaging classifier that detects tumours in X-rays. After
running evasion attacks (FGSM, PGD), you discover the model can be fooled by
imperceptible perturbations. You need defenses that either detect adversarial inputs
before they reach the model or make the model inherently robust to small perturbations.

## The Intuition

Adversarial perturbations exploit two properties: (1) the model's decision boundary is
close to normal inputs, and (2) the perturbations are precisely tuned to the model's
exact weights. Defenses attack one or both of these properties:

- **Adversarial training** pushes the decision boundary further from normal inputs by
  training on perturbed examples.
- **Input randomisation** breaks the precise tuning by adding random noise before
  classification. If the model's output changes dramatically under small random noise,
  the input is likely adversarial.
- **Feature squeezing** quantises the input, destroying the fine-grained perturbation
  that the attack relies on.
- **Gradient regularisation** penalises large input gradients during training, making
  the model's output smoother and harder to attack.

## How It Works

### Adversarial training augmentation

Generate FGSM examples from each training batch and mix them into the training data.
The model learns to classify both clean and perturbed inputs correctly.

### Detection via input randomisation

Add Gaussian noise to the input N times. If the model's output variance exceeds a
threshold, flag the input as adversarial. Legitimate inputs produce stable outputs;
adversarial inputs sit on a knife-edge decision boundary and are sensitive to noise.

### Feature squeezing

Quantise input values to `bit_depth` bits. A perturbation of 0.003 on a [0,1] input
vanishes when rounded to 4-bit precision (1/15 ~ 0.067 resolution).

### Clip and smooth

Clamp inputs to valid range, then apply a moving-average filter. High-frequency
adversarial noise is averaged away.

## In Rust

```rust
use ix_adversarial::defense::{
    adversarial_training_augment,
    input_gradient_regularization,
    detect_adversarial,
    feature_squeezing,
    clip_and_smooth,
};
use ndarray::array;

// Augment training data with adversarial examples
let inputs = vec![array![0.5, 0.5], array![0.3, 0.7]];
let gradients = vec![array![1.0, -1.0], array![-0.5, 0.2]];
let augmented = adversarial_training_augment(&inputs, &gradients, 0.1);

// Gradient regularisation penalty (add to loss during training)
let grad = array![3.0, 4.0];
let penalty = input_gradient_regularization(&grad); // 25.0

// Detect adversarial input via randomisation
let suspicious_input = array![0.5, 0.5];
let is_adversarial = detect_adversarial(
    &suspicious_input,
    |x| x * 100.0,  // model that amplifies inputs
    0.1,             // noise standard deviation
    100,             // number of random samples
    0.01,            // variance threshold
    42,              // RNG seed
);

// Feature squeezing: reduce to 4-bit precision
let squeezed = feature_squeezing(&array![0.503, 0.297, 0.801], 4);

// Clip and smooth: clamp then moving-average
let smoothed = clip_and_smooth(&array![0.0, 1.0, 0.0], 0.0, 1.0, 3);
```

## When To Use This

| Defense | Strength | Cost |
|---------|----------|------|
| **Adversarial training** | Strong against known attacks | Doubles training time; may reduce clean accuracy |
| **Input randomisation** | Detects many attack types | Adds inference latency; false positives possible |
| **Feature squeezing** | Simple; no retraining needed | Reduces input resolution; may hurt fine-grained tasks |
| **Gradient regularisation** | Makes model inherently smoother | Adds a hyperparameter; increases training cost |
| **Clip and smooth** | Removes high-frequency noise | Blurs legitimate detail |

## Pitfalls

1. **Adaptive attacks.** An attacker who knows you use feature squeezing can design
   perturbations that survive quantisation. No single defense is unbreakable.
2. **Detection threshold tuning.** Too sensitive = false positives on noisy but
   legitimate inputs. Too lax = adversarial inputs slip through.
3. **Clean accuracy trade-off.** Adversarial training and gradient regularisation often
   reduce accuracy on unperturbed inputs by 1--3%.

## Going Further

- Combine multiple defenses: squeeze first, then randomise, then classify.
- Use `ix_adversarial::evasion::pgd` to generate strong adversarial examples for
  training augmentation instead of FGSM.
- See [data-poisoning.md](data-poisoning.md) for defenses against training-time attacks.
- See [differential-privacy.md](differential-privacy.md) for protecting model outputs
  against information leakage.
