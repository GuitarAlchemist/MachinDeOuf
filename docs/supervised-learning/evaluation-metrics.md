# Evaluation Metrics

## The Problem

You've trained a fraud detection model and it reports 99% accuracy. Your manager is impressed. But then you realize that only 0.5% of transactions are fraudulent -- a model that blindly predicts "not fraud" for every transaction would achieve 99.5% accuracy. Your "good" model is actually *worse* than doing nothing. Accuracy lied to you.

This happens constantly in real-world machine learning. The metric you choose to evaluate your model determines what you optimize for, and the wrong metric can lead you to deploy a model that fails at the thing you actually care about. A medical screening test that misses 40% of cancer cases but has 95% "accuracy" is dangerous. A spam filter that blocks 10% of legitimate emails is unusable.

Understanding evaluation metrics means understanding the trade-offs between different kinds of errors. Missing a fraudulent transaction (false negative) costs the company money. Blocking a legitimate transaction (false positive) costs customer trust. The right metric depends on which type of error hurts more in your specific application.

## The Intuition

Think of a fire alarm. It can make two types of mistakes:
- **False positive:** The alarm goes off but there's no fire (annoying, but everyone is safe).
- **False negative:** There's a fire but the alarm stays silent (catastrophic).

You'd gladly tolerate some false alarms to ensure the alarm never misses a real fire. That means you care more about **recall** (catching all fires) than **precision** (every alarm being a real fire).

Now think of a courtroom. The system can make two types of mistakes:
- **False positive:** An innocent person is convicted (devastating).
- **False negative:** A guilty person goes free (bad, but less irreversible).

"Innocent until proven guilty" means the justice system prioritizes **precision** (every conviction should be correct) over **recall** (catching every criminal).

Every classification problem has this tension. Evaluation metrics formalize it.

## How It Works

### The Confusion Matrix

Every classification prediction falls into one of four categories:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

In plain English, this means: TP is correctly flagged fraud. TN is correctly allowed legitimate. FP is a legitimate transaction wrongly flagged. FN is fraud that slipped through.

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

In plain English, this means: out of all predictions, what fraction was correct? It treats all errors equally. This is fine when classes are balanced (roughly 50/50), but misleading when one class dominates.

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

In plain English, this means: of everything the model *flagged as positive*, what fraction actually was positive? High precision means few false alarms. When the model says "fraud," you can trust it.

### Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

In plain English, this means: of everything that *actually was positive*, what fraction did the model catch? High recall means few missed cases. The model finds most of the real fraud.

### F1 Score

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

In plain English, this means: the harmonic mean of precision and recall. It's a single number that balances both concerns. An F1 of 0.8 means neither precision nor recall is terrible. The harmonic mean punishes extreme imbalances -- if precision is 1.0 but recall is 0.1, F1 is only 0.18, not 0.55 like the arithmetic mean would suggest.

### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

In plain English, this means: for regression problems, measure the average squared difference between predicted and actual values. Squaring penalizes large errors disproportionately -- being off by $100K is much worse than being off by $10K, and MSE reflects that.

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

In plain English, this means: take the square root of MSE to get the error back into the original units. If you're predicting house prices in dollars, RMSE is in dollars. An RMSE of $25,000 means "on average, we're about $25K off."

### R-squared (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

In plain English, this means: how much better is the model than just predicting the average every time? R^2 = 1.0 means perfect predictions. R^2 = 0.0 means the model is no better than the mean. R^2 < 0 means the model is actively worse than the mean.

## In Rust

### Classification Metrics

```rust
use ndarray::array;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Fraud detection results
    //         actual:  fraud, fraud, legit, legit, fraud, legit, legit, fraud
    let y_true = array![1,     1,     0,     0,     1,     0,     0,     1];
    //       predicted:  fraud, legit, legit, fraud, fraud, legit, legit, legit
    let y_pred = array![1,     0,     0,     1,     1,     0,     0,     0];

    // Overall accuracy: 6/8 = 0.75
    println!("Accuracy:  {:.4}", accuracy(&y_true, &y_pred));

    // Precision for fraud (class 1): TP=2, FP=1 -> 2/3 = 0.667
    // "When we flag fraud, we're right 67% of the time"
    println!("Precision (fraud): {:.4}", precision(&y_true, &y_pred, 1));

    // Recall for fraud (class 1): TP=2, FN=2 -> 2/4 = 0.50
    // "We catch 50% of actual fraud"
    println!("Recall (fraud):    {:.4}", recall(&y_true, &y_pred, 1));

    // F1 for fraud: 2 * 0.667 * 0.5 / (0.667 + 0.5) = 0.571
    println!("F1 (fraud):        {:.4}", f1_score(&y_true, &y_pred, 1));

    // Metrics for the legitimate class (class 0)
    println!("Precision (legit): {:.4}", precision(&y_true, &y_pred, 0));
    println!("Recall (legit):    {:.4}", recall(&y_true, &y_pred, 0));
    println!("F1 (legit):        {:.4}", f1_score(&y_true, &y_pred, 0));
}
```

### Regression Metrics

```rust
use ndarray::array;
use ix_supervised::metrics::{mse, rmse, r_squared};

fn main() {
    // House price predictions (in thousands of dollars)
    let y_true = array![250.0, 300.0, 180.0, 420.0, 350.0];
    let y_pred = array![260.0, 285.0, 195.0, 400.0, 365.0];

    println!("MSE:  {:.2}", mse(&y_true, &y_pred));      // in (thousands $)^2
    println!("RMSE: {:.2}", rmse(&y_true, &y_pred));      // in thousands $
    println!("R^2:  {:.4}", r_squared(&y_true, &y_pred)); // unitless, 0 to 1
}
```

### Combining with a Classifier

```rust
use ndarray::array;
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    let x_train = array![
        [0.0, 0.0], [0.5, 0.5], [0.3, 0.2],
        [3.0, 3.0], [3.5, 3.5], [3.2, 3.3],
    ];
    let y_train = array![0, 0, 0, 1, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iterations(1000);
    model.fit(&x_train, &y_train);

    let y_pred = model.predict(&x_train);

    println!("Accuracy:       {:.4}", accuracy(&y_train, &y_pred));
    println!("Precision (1):  {:.4}", precision(&y_train, &y_pred, 1));
    println!("Recall (1):     {:.4}", recall(&y_train, &y_pred, 1));
    println!("F1 (1):         {:.4}", f1_score(&y_train, &y_pred, 1));
}
```

## When To Use This

| Metric | Use When | Do NOT Use When |
|---|---|---|
| **Accuracy** | Classes are balanced (50/50 split) | Classes are imbalanced (99/1 split) |
| **Precision** | False positives are costly (spam filter: don't block good email) | Missing positives is the bigger risk |
| **Recall** | False negatives are costly (cancer screening: don't miss a tumor) | False alarms are the bigger risk |
| **F1 Score** | You need a single balanced metric; both errors matter | One error type is clearly more important than the other |
| **MSE** | Regression; large errors are disproportionately bad | You want errors in original units |
| **RMSE** | Regression; you want interpretable error in original units | Outliers dominate and you want robustness |
| **R-squared** | You want to know "how much variance does my model explain?" | Comparing models across different datasets (R^2 is dataset-dependent) |

### Decision Guide for Imbalanced Classification

| Scenario | Primary Metric | Secondary Metric |
|---|---|---|
| Fraud detection (rare fraud, costly misses) | Recall | F1 |
| Spam filter (annoying false positives) | Precision | F1 |
| Medical screening (must not miss disease) | Recall | Precision (to limit unnecessary follow-ups) |
| Search engine results (relevance matters) | Precision | Recall |
| Balanced binary classification | F1 or Accuracy | Precision, Recall |

## Key Parameters

All metric functions take `Array1` inputs and return `f64`:

| Function | Signature | Domain |
|---|---|---|
| `accuracy` | `(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64` | Classification |
| `precision` | `(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64` | Classification |
| `recall` | `(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64` | Classification |
| `f1_score` | `(y_true: &Array1<usize>, y_pred: &Array1<usize>, class: usize) -> f64` | Classification |
| `mse` | `(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64` | Regression |
| `rmse` | `(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64` | Regression |
| `r_squared` | `(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64` | Regression |

Note that `precision`, `recall`, and `f1_score` require a `class` parameter specifying which class to treat as the "positive" class. To evaluate a binary classifier for fraud, pass `class: 1` (fraud) to get precision/recall for fraud detection.

## Pitfalls

**Accuracy on imbalanced data is meaningless.** If 99.5% of transactions are legitimate, a model that always predicts "legitimate" gets 99.5% accuracy but catches zero fraud. Always check precision and recall when classes are imbalanced.

**Precision and recall are inversely related.** Lowering the classification threshold (e.g., from 0.5 to 0.3) catches more positives (higher recall) but also flags more negatives (lower precision). You can't maximize both simultaneously -- you have to decide which matters more for your application.

**Per-class metrics are essential.** Overall accuracy can mask poor performance on the minority class. Always compute precision, recall, and F1 *for each class individually*.

**R-squared can be negative.** This isn't a bug -- it means the model's predictions are worse than just predicting the mean. If you see a negative R-squared, your model has a fundamental problem.

**MSE is in squared units.** MSE of 625 for house prices in thousands means RMSE of 25, which means "roughly $25K off on average." Always take the square root for interpretability.

**Edge cases.** If the model never predicts a class, precision for that class is 0/0. The ix implementation returns 0.0 in this case, which is a reasonable convention but worth knowing about.

## Going Further

- **Threshold tuning:** Most classifiers produce probabilities. By default, the threshold is 0.5, but you can adjust it. Lowering it to 0.3 boosts recall at the expense of precision. Plot the precision-recall curve at different thresholds to find the right trade-off.
- **Macro vs. micro averaging:** For multi-class problems, you can compute F1 per class and average (macro), or pool all TP/FP/FN across classes (micro). Macro treats each class equally; micro weights by class frequency.
- **Cross-validation:** Instead of a single train/test split, use k-fold cross-validation to get more reliable metric estimates. Train k models, each tested on a different fold, and average the metrics.
- **Confusion matrix visualization:** While ix doesn't include visualization, the raw TP/FP/TN/FN counts can be computed from the `accuracy`, `precision`, and `recall` functions for any post-hoc analysis.
- **Regression alternatives:** Mean Absolute Error (MAE) is more robust to outliers than MSE/RMSE. It can be computed as `(y_true - y_pred).mapv(f64::abs).mean().unwrap()`.
