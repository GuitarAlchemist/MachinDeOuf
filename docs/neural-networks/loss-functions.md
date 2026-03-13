# Loss Functions

> The number that tells your model how wrong it is — and choosing the right one changes everything.

## The Problem

Your neural network makes a prediction. The prediction is wrong. But *how* wrong? And how should you quantify "wrong" so the model can improve?

The loss function is your model's report card. It takes the prediction and the true answer, and outputs a single number: low = good, high = bad. Gradient descent minimizes this number. **Different loss functions define different notions of "wrong"** — and choosing poorly can silently sabotage your model.

## The Intuition

### MSE: Average Squared Error

**Mean Squared Error** — the default for regression (predicting numbers).

Think of a dartboard. MSE measures how far your darts land from the bullseye, on average. But squaring the distances means a dart that's 10cm off is punished **100 times more** than a dart that's 1cm off. This makes MSE very sensitive to outliers — one terrible prediction dominates the loss.

### Cross-Entropy: Measuring Surprise

**Binary Cross-Entropy** — the default for classification (predicting categories).

Think of a weather forecaster. If they say "90% chance of rain" and it rains, that's unsurprising — low loss. If they say "10% chance of rain" and it rains, that's very surprising — high loss. Cross-entropy measures how "surprised" your model is by the true answer.

The key difference from MSE: cross-entropy cares about *confidence*. Predicting 0.51 for a true positive and 0.99 for a true positive have the same sign but very different losses. Cross-entropy rewards confident correct predictions and heavily penalizes confident wrong predictions.

## How It Works

### Mean Squared Error (MSE)

`MSE = (1/n) × Σᵢ (yᵢ - ŷᵢ)²`

In plain English: for each prediction, compute the error (true - predicted), square it, and average over all samples.

**Gradient**: `dMSE/dŷ = (2/n) × (ŷ - y)`

In plain English: the gradient is proportional to the error. Larger errors get larger gradients, pushing the model harder to fix them.

**When to use**: Regression — predicting continuous values (price, temperature, score).

### Binary Cross-Entropy (BCE)

`BCE = -(1/n) × Σᵢ [yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)]`

In plain English: for each sample, if the true label is 1, penalize based on -log(predicted probability of 1). If the true label is 0, penalize based on -log(predicted probability of 0). Average over all samples.

Why logarithm? Because -log(0.99) ≈ 0.01 (confident and correct → tiny loss) while -log(0.01) ≈ 4.6 (confident and wrong → huge loss). This creates a steep gradient when the model is confidently wrong, accelerating correction.

**Gradient**: `dBCE/dŷ = (1/n) × (ŷ - y) / (ŷ × (1 - ŷ))`

**When to use**: Binary classification — predicting yes/no, spam/not-spam, fraud/legitimate.

### Categorical Cross-Entropy

`CCE = -(1/n) × Σᵢ Σ_c y_ic × log(ŷ_ic)`

In plain English: for multi-class problems. y is one-hot encoded (e.g., [0, 0, 1, 0] for class 2). Only the true class's predicted probability contributes to the loss.

**When to use**: Multi-class classification — digit recognition (0-9), species classification, etc.

## In Rust

```rust
use ndarray::array;
use machin_nn::loss;

// --- Regression: MSE ---
let predicted = array![[2.5], [0.0], [2.1], [7.8]];
let actual    = array![[3.0], [0.0], [2.0], [8.0]];

let mse = loss::mse_loss(&predicted, &actual);
println!("MSE: {:.4}", mse);  // Small — predictions are close

let mse_grad = loss::mse_gradient(&predicted, &actual);
println!("MSE gradient: {:?}", mse_grad);
// Gradient points from predicted toward actual

// --- Classification: Binary Cross-Entropy ---
let pred_probs = array![[0.9], [0.2], [0.8], [0.1]];  // Model's probabilities
let true_labels = array![[1.0], [0.0], [1.0], [0.0]];  // Actual classes

let bce = loss::binary_cross_entropy(&pred_probs, &true_labels);
println!("BCE: {:.4}", bce);  // Low — model is confident and correct

let bce_grad = loss::binary_cross_entropy_gradient(&pred_probs, &true_labels);
println!("BCE gradient: {:?}", bce_grad);

// --- What happens with a confident WRONG prediction? ---
let bad_pred = array![[0.01]];   // Model says 1% chance of class 1
let true_val = array![[1.0]];    // But it IS class 1!

let bad_loss = loss::binary_cross_entropy(&bad_pred, &true_val);
println!("Confident wrong: {:.4}", bad_loss);  // Very high!
```

### Using Loss Functions in Training

```rust
use ndarray::array;
use machin_nn::layer::{Dense, Layer};
use machin_nn::loss;

let mut layer = Dense::new(3, 1);
let x = array![[1.0, 2.0, 3.0]];
let target = array![[5.0]];

// Forward
let pred = layer.forward(&x);
let loss_val = loss::mse_loss(&pred, &target);

// Backward — gradient flows from loss through layers
let grad = loss::mse_gradient(&pred, &target);
layer.backward(&grad, 0.01);

println!("Loss before update: {:.4}", loss_val);

// After update, loss should be lower
let new_pred = layer.forward(&x);
let new_loss = loss::mse_loss(&new_pred, &target);
println!("Loss after update: {:.4}", new_loss);
```

## When To Use This

| Loss Function | Task | Output Activation | Gradient Behavior |
|--------------|------|-------------------|-------------------|
| **MSE** | Regression (predict a number) | Linear (none) | Proportional to error |
| **Binary Cross-Entropy** | Binary classification | Sigmoid | Strong gradient when confidently wrong |
| **Categorical Cross-Entropy** | Multi-class classification | Softmax | Same as BCE but for multiple classes |

**Quick decision guide:**
- Predicting a continuous value? → **MSE**
- Predicting yes/no? → **Binary Cross-Entropy**
- Predicting one of N classes? → **Categorical Cross-Entropy**

## Key Parameters

Loss functions themselves have no hyperparameters — they're fixed formulas. But be aware of:

| Concern | What To Watch |
|---------|---------------|
| Numerical stability | Cross-entropy with log(0) = -infinity. MachinDeOuf clamps predictions to avoid this. |
| Scale | MSE is in squared units. RMSE (√MSE) is in original units and often more interpretable. |
| Outlier sensitivity | MSE is very sensitive to outliers. Consider Huber loss (not yet in MachinDeOuf) for robust regression. |

## Pitfalls

- **Don't use MSE for classification.** MSE treats the difference between 0.4 and 0.6 the same as between 0.0 and 0.2. Cross-entropy correctly penalizes confident wrong predictions much more, producing better gradients.
- **Don't use cross-entropy for regression.** Cross-entropy expects probabilities (0 to 1). Raw regression outputs can be any number.
- **Watch for log(0).** If your model predicts exactly 0 or 1, log(0) = -infinity. MachinDeOuf clamps predictions to a small epsilon range to prevent this, but if you implement custom loss, be careful.
- **Check output activation.** Binary cross-entropy assumes predictions are probabilities (sigmoid output). If your output layer has no activation, the predictions might be negative or > 1, making the loss meaningless.
- **Loss going to NaN?** Usually means: learning rate too high, log(0) somewhere, or exploding gradients. Lower the learning rate first.

## Going Further

- **Before**: [Perceptron to MLP](perceptron-to-mlp.md) — the networks these losses train
- **Before**: [Backpropagation](backpropagation.md) — how gradients of the loss propagate back
- **Related**: [Evaluation Metrics](../supervised-learning/evaluation-metrics.md) — accuracy, precision, recall (different from loss!)
- **Related**: [Gradient Descent](../optimization/gradient-descent.md) — the optimizer that minimizes the loss
