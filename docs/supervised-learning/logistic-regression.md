# Logistic Regression

## The Problem

Your email provider processes millions of messages per hour. Each message has measurable features: how many times the word "free" appears, whether the sender is in the contact list, the ratio of links to text, the presence of certain header patterns. The system needs to decide, for every single email, whether it is spam (class 1) or not spam (class 0).

This is not a continuous prediction -- you need a yes-or-no answer. But you also want to know *how confident* the model is. An email that is 51% likely to be spam should perhaps land in a "suspicious" folder, while one at 99% goes straight to the trash. You need a model that outputs a probability between 0 and 1.

Logistic regression does exactly this. Despite its name, it is a classification algorithm. It takes the linear combination of features (just like linear regression), then squashes the result through a sigmoid function to produce a probability. Training adjusts the weights to maximize the likelihood that the model's probabilities match the actual labels.

## The Intuition

Think of logistic regression as linear regression with a translator bolted onto the end. Linear regression can output any number from negative infinity to positive infinity. The sigmoid function acts as a translator that converts that raw number into a value between 0 and 1 -- a probability.

Imagine a thermometer where the mercury can go as high or as low as it wants. Now put that thermometer behind a special lens that compresses the reading: very negative numbers get pushed toward 0, very positive numbers get pushed toward 1, and the middle range spreads out smoothly between them. That S-shaped compression is the sigmoid.

During training, the model adjusts its weights so that spam emails get a high raw score (which the sigmoid maps to near 1.0) and legitimate emails get a low raw score (mapped to near 0.0). The learning process nudges the weights a little bit at a time in the direction that makes the probabilities more accurate -- this is gradient descent.

## How It Works

### Step 1: Compute the raw score

$$
z = \mathbf{X} \mathbf{w} + b
$$

In plain English, this means: for each email, multiply its features by the learned weights, add them up, and add a bias term. This gives a raw score that can be any real number.

### Step 2: Apply the sigmoid function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

In plain English, this means: squeeze the raw score into the range (0, 1). Large positive scores become probabilities near 1 (likely spam). Large negative scores become probabilities near 0 (likely not spam). A score of exactly 0 maps to 0.5 -- total uncertainty.

### Step 3: Compute the log loss (binary cross-entropy)

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(a_i) + (1 - y_i) \log(1 - a_i) \right]
$$

In plain English, this means: measure how wrong the model is. If the true label is 1 (spam) and the model says probability 0.01, the loss is huge. If it says 0.99, the loss is tiny. This function harshly penalizes confident wrong answers.

### Step 4: Compute gradients and update weights

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^T (\mathbf{a} - \mathbf{y})
$$

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}
$$

In plain English, this means: figure out which direction to nudge each weight to reduce the loss, then take a small step in that direction. The size of the step is controlled by the learning rate alpha. Repeat this many times until the weights converge.

### Step 5: Classify

Apply a threshold (default 0.5) to the probability:

$$
\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

In plain English, this means: if the model thinks the email is more likely spam than not, label it spam.

## In Rust

```rust
use ndarray::array;
use machin_supervised::logistic_regression::LogisticRegression;
use machin_supervised::traits::Classifier;
use machin_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [word_free_count, has_known_sender (0/1), link_ratio, suspicious_headers]
    let x = array![
        [0.0, 1.0, 0.05, 0.0],  // legitimate
        [0.0, 1.0, 0.10, 0.0],  // legitimate
        [1.0, 0.0, 0.02, 0.0],  // legitimate (one "free" but known pattern)
        [5.0, 0.0, 0.80, 1.0],  // spam
        [3.0, 0.0, 0.60, 1.0],  // spam
        [7.0, 0.0, 0.90, 1.0],  // spam
    ];
    let y = array![0, 0, 0, 1, 1, 1]; // 0 = not spam, 1 = spam

    // Build and train the model
    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iterations(1000);
    model.fit(&x, &y);

    // Predict hard labels
    let predictions = model.predict(&x);
    println!("Predictions: {}", predictions);

    // Get probability estimates (columns: [P(class=0), P(class=1)])
    let probabilities = model.predict_proba(&x);
    println!("Spam probabilities: {}", probabilities.column(1));

    // Evaluate
    println!("Accuracy:  {:.4}", accuracy(&y, &predictions));
    println!("Precision: {:.4}", precision(&y, &predictions, 1));
    println!("Recall:    {:.4}", recall(&y, &predictions, 1));
    println!("F1:        {:.4}", f1_score(&y, &predictions, 1));
}
```

## When To Use This

| Situation | Logistic Regression | Alternative | Why |
|---|---|---|---|
| Binary classification, linear decision boundary | Yes | -- | Fast, interpretable, probabilistic |
| Need probability outputs, not just labels | Yes | -- | Sigmoid naturally outputs calibrated probabilities |
| Non-linear decision boundaries | No | Decision tree, SVM with kernel, neural net | Logistic regression can only draw straight lines |
| Multi-class (3+ classes) | Limited | Naive Bayes, KNN, decision tree | MachinDeOuf's implementation is binary only |
| Very high-dimensional sparse data (NLP) | Yes | Naive Bayes | Logistic regression handles sparse features well |
| Need to understand feature importance | Yes | -- | Weight magnitude and sign are directly interpretable |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.01` | Step size for gradient descent. Too high: oscillates. Too low: converges slowly. |
| `max_iterations` | `1000` | Number of gradient descent passes over the full dataset. |

Configure via the builder pattern:
```rust
LogisticRegression::new()
    .with_learning_rate(0.05)
    .with_max_iterations(2000)
```

## Pitfalls

**Learning rate too high.** The loss will oscillate or diverge instead of decreasing. If predictions are random after training, try reducing the learning rate by a factor of 10.

**Features not scaled.** Logistic regression uses gradient descent, which is sensitive to feature scale. If one feature ranges from 0 to 1 and another from 0 to 10,000, the gradients will be dominated by the large-scale feature. Normalize or standardize features before training.

**Linearly inseparable data.** If spam and legitimate emails overlap in feature space with no straight line separating them, logistic regression will do its best but the accuracy ceiling will be low. Consider adding polynomial feature interactions or switching to a non-linear model.

**Class imbalance.** If 99% of emails are legitimate and 1% are spam, the model may learn to always predict "not spam" and still achieve 99% accuracy. Use precision, recall, and F1 (see [evaluation-metrics.md](./evaluation-metrics.md)) instead of accuracy, and consider adjusting the decision threshold away from 0.5.

**Binary only.** The MachinDeOuf implementation supports two classes (0 and 1). For multi-class problems, use one-vs-rest with multiple logistic regression models, or choose a different classifier.

## Going Further

- **Feature engineering:** Add interaction terms (e.g., `word_free_count * link_ratio`) to capture non-linear patterns within a linear framework.
- **Regularization:** Combine with `machin_optimize` to add L2 (Ridge) or L1 (Lasso) penalty terms to prevent overfitting.
- **Threshold tuning:** Instead of the default 0.5, pick a threshold that optimizes the F1 score or minimizes false negatives, depending on your application.
- **Evaluation deep dive:** See [evaluation-metrics.md](./evaluation-metrics.md) for when to prefer precision over recall.
