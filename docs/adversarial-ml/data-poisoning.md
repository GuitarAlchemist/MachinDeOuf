# Data Poisoning Detection

## The Problem

You are training a spam classifier on user-reported examples. An adversary who controls
a fraction of the reports can inject mislabelled samples -- legitimate emails labelled as
spam, or spam labelled as legitimate -- to corrupt the model. You need automated methods
to detect and remove these poisoned samples before they degrade your classifier.

Data poisoning also threatens medical ML (tampered radiology labels), autonomous driving
(corrupted lidar annotations), and any system trained on crowd-sourced or scraped data.

## The Intuition

Poisoned samples are imposters: they have the features of one class but the label of
another. Detection methods look for samples that "do not belong" with their labelled
neighbours:

- **KNN label consistency:** If a sample's K nearest neighbours mostly disagree with its
  label, it is suspicious. A dog photo labelled "cat" will be surrounded by other dog
  photos.
- **Influence functions:** Estimate how much removing a single training sample would
  change the model's prediction on a test point. Poisoned samples have outsized influence.
- **Spectral signatures:** Backdoor attacks embed a consistent pattern across poisoned
  samples. This pattern shows up as an outlier direction in the feature covariance. Power
  iteration finds this direction, and samples with large projections onto it are flagged.

## How It Works

### KNN label flip detection

For each sample, find its K nearest neighbours by Euclidean distance. If the majority
label among neighbours differs from the sample's own label, flag it.

### Influence function (simplified)

Approximate how much a training point affects a test prediction using the gradient and
Hessian of the loss. High-influence points may be poisoned.

### Spectral signature defense

Per class, centre the features, find the top singular vector via power iteration, and
flag samples whose projection onto that vector exceeds a percentile threshold.

## In Rust

```rust
use machin_adversarial::poisoning::{
    detect_label_flips,
    influence_function,
    spectral_signature_defense,
};
use ndarray::{array, Array2};

// KNN-based label flip detection
let features = Array2::from_shape_vec((6, 2), vec![
    0.0, 0.0,  0.1, 0.1,  0.05, 0.05,   // cluster 0
    1.0, 1.0,  1.1, 1.1,  1.05, 1.05,   // cluster 1
]).unwrap();
let labels = array![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // index 2 is flipped!
let suspicious = detect_label_flips(&features, &labels, 3);
assert!(suspicious.contains(&2));

// Influence function
let test_point = array![0.5, 0.5];
let influences = influence_function(&features, &labels, &test_point, 1.0, 0.1);

// Spectral signature defense
let flagged = spectral_signature_defense(&features, &labels, 2, 50.0);
```

## When To Use This

| Method | Detects | Cost |
|--------|---------|------|
| **KNN label flips** | Mislabelled samples near wrong cluster | O(N^2) pairwise distances |
| **Influence functions** | Points with outsized impact on predictions | Requires Hessian approximation |
| **Spectral signatures** | Backdoor patterns (consistent trigger) | Power iteration per class |

## Pitfalls

1. **KNN struggles at decision boundaries.** Samples near the true boundary between
   classes may be falsely flagged because their neighbours are mixed.
2. **Influence functions are approximate.** The linear approximation is only valid near
   the optimum; early in training it can be misleading.
3. **Spectral defense assumes the poison is a minority.** If more than ~30% of a class
   is poisoned, the "outlier" direction may actually be the majority.

## Going Further

- Run `detect_label_flips` as a pre-processing step before training, removing flagged
  samples or sending them for human review.
- Combine with adversarial training from `machin_adversarial::defense` to build models
  robust to both evasion and poisoning attacks.
- Use `machin_unsupervised` clustering (K-Means) to independently verify class structure
  before training.
- See [differential-privacy.md](differential-privacy.md) for limiting the influence of
  any single training sample via DP-SGD.
