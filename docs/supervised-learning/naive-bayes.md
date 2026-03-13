# Naive Bayes

## The Problem

You're building a product review system for an e-commerce platform. Customers leave text reviews, and you need to automatically classify each review as positive (class 1) or negative (class 0). After converting each review into numerical features -- perhaps the average word embedding values, the review length, the count of exclamation marks, and a sentiment lexicon score -- you need a classifier that trains fast, handles the data well even with limited examples, and produces probability estimates.

Sentiment analysis is a domain where classes often overlap substantially in feature space. A review mentioning "not bad" has the word "bad" but means something positive. You need a model that weighs all features together using probability theory rather than drawing hard decision boundaries.

Naive Bayes is one of the oldest and most reliable classifiers for this kind of task. It applies Bayes' theorem to compute the probability of each class given the observed features, making the simplifying assumption that features are independent given the class. This "naive" assumption is almost never true in practice, yet the classifier works remarkably well anyway.

## The Intuition

Imagine you're a doctor diagnosing whether a patient has a cold or the flu. You observe symptoms: temperature, cough severity, and fatigue level. For each disease, you know the typical range of each symptom from past patients. Bayes' theorem lets you combine these observations: "Given this temperature AND this cough AND this fatigue, which disease is more probable?"

The "naive" part is the independence assumption. In reality, high temperature and severe cough are correlated -- but Naive Bayes pretends they're not. It treats each symptom as providing independent evidence. This simplification makes the math tractable and, surprisingly, rarely hurts classification accuracy much.

The Gaussian variant used in MachinDeOuf assumes that each feature follows a bell curve (normal distribution) within each class. So for "positive reviews," the sentiment score might have a mean of 0.7 with some spread, while for "negative reviews," it might center around 0.3. When a new review comes in, the model asks: "Is this sentiment score more likely under the positive bell curve or the negative one?" -- and does this for every feature, then multiplies the answers together.

## How It Works

### Step 1: Estimate class priors

$$
P(c) = \frac{\text{number of training samples in class } c}{n}
$$

In plain English, this means: count how many positive and negative reviews you have. If 60% of reviews are positive, the prior probability of "positive" is 0.6. Before looking at any features, the model already leans toward predicting the more common class.

### Step 2: Estimate per-class Gaussian parameters

For each class c and each feature j, compute the mean and variance:

$$
\mu_{c,j} = \frac{1}{n_c} \sum_{i \in c} x_{i,j}
$$

$$
\sigma^2_{c,j} = \frac{1}{n_c} \sum_{i \in c} (x_{i,j} - \mu_{c,j})^2
$$

In plain English, this means: for each class, learn what "typical" looks like for each feature. Positive reviews might have an average sentiment score of 0.7 with a variance of 0.04. This defines a bell curve that describes how that feature behaves in that class.

### Step 3: Compute class-conditional likelihoods using the Gaussian PDF

$$
P(x_j \mid c) = \frac{1}{\sqrt{2\pi \sigma^2_{c,j}}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma^2_{c,j}}\right)
$$

In plain English, this means: for a new review's feature value, ask "how likely is this value under the bell curve for class c?" A sentiment score of 0.8 is very likely under the "positive" bell curve but unlikely under the "negative" one.

### Step 4: Apply Bayes' theorem with the naive independence assumption

$$
P(c \mid x) \propto P(c) \prod_{j=1}^{p} P(x_j \mid c)
$$

In plain English, this means: multiply the prior (how common is this class?) by the likelihood of each feature (how typical is each feature value for this class?), assuming features contribute independently. The class with the highest product wins.

### Step 5: Normalize to get probabilities

In practice, we work in log space to avoid numerical underflow (multiplying many small probabilities), then convert back via the log-sum-exp trick.

In plain English, this means: the raw products are not proper probabilities (they don't sum to 1). We rescale them so they do, giving us a clean probability distribution over classes.

## In Rust

```rust
use ndarray::array;
use machin_supervised::naive_bayes::GaussianNaiveBayes;
use machin_supervised::traits::Classifier;
use machin_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [sentiment_score, review_length_norm, exclamation_count, avg_word_embedding]
    // Labels: 0 = negative, 1 = positive
    let x = array![
        [0.8,  0.6, 2.0, 0.55],   // positive
        [0.7,  0.4, 1.0, 0.50],   // positive
        [0.9,  0.7, 3.0, 0.60],   // positive
        [0.65, 0.5, 1.0, 0.48],   // positive
        [0.2,  0.3, 0.0, 0.30],   // negative
        [0.1,  0.8, 0.0, 0.25],   // negative
        [0.3,  0.4, 0.0, 0.35],   // negative
        [0.15, 0.6, 0.0, 0.28],   // negative
    ];
    let y = array![1, 1, 1, 1, 0, 0, 0, 0];

    // Train the classifier
    let mut model = GaussianNaiveBayes::new();
    model.fit(&x, &y);

    // Predict on new reviews
    let new_reviews = array![
        [0.75, 0.5, 2.0, 0.52],   // probably positive
        [0.25, 0.4, 0.0, 0.32],   // probably negative
    ];

    let predictions = model.predict(&new_reviews);
    println!("Predictions: {}", predictions);

    // Get probability estimates
    let proba = model.predict_proba(&new_reviews);
    for i in 0..proba.nrows() {
        println!(
            "Review {}: P(negative)={:.3}, P(positive)={:.3}",
            i, proba[[i, 0]], proba[[i, 1]]
        );
    }

    // Evaluate on training data
    let train_pred = model.predict(&x);
    println!("Accuracy:  {:.4}", accuracy(&y, &train_pred));
    println!("F1 (positive): {:.4}", f1_score(&y, &train_pred, 1));
}
```

## When To Use This

| Situation | Naive Bayes | Alternative | Why |
|---|---|---|---|
| Small training set | Yes | -- | Needs very few examples to estimate means and variances |
| Fast training and prediction needed | Yes | -- | Training is O(n*p), prediction is O(C*p). No iteration. |
| Text/NLP classification | Yes | Logistic regression | Classic choice for document classification |
| Features are genuinely independent | Yes | -- | The naive assumption is actually correct |
| Features are highly correlated | Maybe | Logistic regression, SVM | Still works surprisingly well, but probabilities may be poorly calibrated |
| Need well-calibrated probabilities | No | Logistic regression | Naive independence assumption distorts probability estimates |
| Complex non-linear boundaries | No | Decision tree, random forest | Naive Bayes assumes simple Gaussian class distributions |

## Key Parameters

`GaussianNaiveBayes::new()` takes no configuration parameters. The model learns everything from the data:

| Learned Parameter | Description |
|---|---|
| `means` | Per-class mean for each feature |
| `variances` | Per-class variance for each feature (with epsilon floor of 1e-9 to avoid division by zero) |
| `priors` | Prior probability of each class |

This is one of the simplest models to use -- there is nothing to tune.

## Pitfalls

**The Gaussian assumption.** The model assumes each feature follows a bell curve within each class. If a feature is bimodal, heavily skewed, or binary, the Gaussian assumption is a poor fit. Binary features are better served by Bernoulli Naive Bayes (not currently in MachinDeOuf).

**Correlated features degrade probability estimates.** While classification accuracy is often robust to the independence violation, the *probabilities* can be wildly overconfident or underconfident. If you need calibrated probabilities, use logistic regression or apply post-hoc calibration.

**Zero variance features.** If a feature has exactly the same value for all samples in a class, the variance is zero and the Gaussian PDF becomes a delta function. MachinDeOuf guards against this with a minimum variance of 1e-9, but such features should be removed.

**Continuous features only.** The Gaussian variant expects continuous numerical features. Categorical features must be encoded numerically (e.g., one-hot encoding) before use, and the Gaussian assumption on binary features is a rough approximation.

**Dominated by informative features.** Unlike decision trees that can ignore irrelevant features, Naive Bayes multiplies probabilities from *all* features. Noisy, irrelevant features dilute the signal from informative ones. Feature selection helps.

## Going Further

- **Probability foundations:** The `machin_math::stats` module provides mean, variance, and other statistical functions used internally by this classifier.
- **Text classification pipeline:** Convert text to numerical features using TF-IDF or word embeddings, then feed into `GaussianNaiveBayes`. For binary word-presence features, a Bernoulli variant would be more appropriate.
- **Multi-class:** The implementation handles any number of classes automatically -- labels 0, 1, 2, ... are all supported.
- **Ensemble combination:** Use Naive Bayes predictions as a feature input to a `RandomForest` for a stacking ensemble approach.
- **Evaluation:** See [evaluation-metrics.md](./evaluation-metrics.md) for understanding when accuracy, precision, recall, and F1 are appropriate for sentiment analysis tasks.
