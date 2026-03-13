# Probability & Statistics

> The mathematical language of uncertainty — how to describe, measure, and reason about data.

## The Problem

You're building a spam filter. Out of 10,000 emails, 2,000 are spam. A new email arrives containing the word "lottery." How likely is it spam?

This question is impossible to answer without probability and statistics. Every ML algorithm either *uses* probability directly (Naive Bayes, logistic regression, bandits) or *relies on statistics* to evaluate its performance (mean error, variance, confidence intervals). You need both.

## The Intuition

### Statistics: Describing What You Have

Statistics answers: "What does my data look like?"

Imagine measuring the heights of 100 people. You can't remember all 100 numbers, so you summarize:

- **Mean (average)**: The "center" of your data. If heights are [170, 175, 180], the mean is 175.
- **Variance**: How spread out the data is. If everyone is 175cm, variance is 0. If heights range from 150 to 200, variance is large.
- **Standard deviation**: The square root of variance. It's in the same units as your data, so it's more intuitive. "Heights vary by about 10cm" is more useful than "variance is 100."
- **Median**: The middle value when sorted. Less sensitive to outliers than the mean — one 300cm person doesn't wreck your summary.

### Probability: Reasoning About What Could Happen

Probability answers: "How likely is this?"

- **P(spam) = 0.2** means 20% of emails are spam (based on your dataset).
- **P(lottery | spam) = 0.8** means 80% of spam emails contain "lottery."
- **P(lottery | not spam) = 0.01** means only 1% of legitimate emails contain "lottery."

### Bayes' Theorem: Flipping the Question

You know P(lottery | spam) — the probability of seeing "lottery" *given* the email is spam. But you want P(spam | lottery) — the probability the email is spam *given* you see "lottery."

Bayes' theorem flips this:

```
P(spam | lottery) = P(lottery | spam) × P(spam) / P(lottery)
```

In plain English: "Update your initial belief (P(spam) = 0.2) using the new evidence (the word 'lottery')."

Working through the numbers:
- P(lottery) = P(lottery|spam) × P(spam) + P(lottery|not spam) × P(not spam)
- P(lottery) = 0.8 × 0.2 + 0.01 × 0.8 = 0.168
- P(spam | lottery) = 0.8 × 0.2 / 0.168 = 0.952

That email is 95.2% likely to be spam. Bayes' theorem is the foundation of Naive Bayes classifiers, Bayesian inference, and probabilistic reasoning throughout ML.

## How It Works

### Descriptive Statistics

**Mean** (expected value):

`μ = (1/n) × Σᵢ xᵢ`

In plain English: add all values, divide by how many there are.

**Variance** (population):

`σ² = (1/n) × Σᵢ (xᵢ - μ)²`

In plain English: how far each value is from the mean, on average. Squaring ensures positive and negative deviations don't cancel out.

**Sample variance** (Bessel's correction):

`s² = (1/(n-1)) × Σᵢ (xᵢ - μ)²`

In plain English: when estimating variance from a sample (not the full population), divide by n-1 instead of n. This corrects a subtle bias — samples tend to underestimate the true spread.

**Standard deviation**:

`σ = √(σ²)`

In plain English: the "typical" distance from the mean, in the original units.

**Covariance** (between two variables):

`cov(X, Y) = (1/n) × Σᵢ (xᵢ - μₓ)(yᵢ - μᵧ)`

In plain English: do X and Y move together? Positive covariance means when X goes up, Y tends to go up. Negative means they move oppositely.

**Correlation** (normalized covariance):

`cor(X, Y) = cov(X, Y) / (σₓ × σᵧ)`

In plain English: covariance scaled to [-1, 1]. +1 means perfect positive relationship, -1 means perfect negative, 0 means no linear relationship.

### Probability Distributions

A distribution describes what values a random variable can take and how likely each is.

**Normal (Gaussian) distribution**: The bell curve. Described by mean μ and standard deviation σ. Many natural phenomena are roughly normal (heights, measurement errors). Central limit theorem: averages of many random things tend toward normal.

**Uniform distribution**: All values equally likely. Rolling a fair die. Random initialization of model weights.

**Bernoulli distribution**: Two outcomes (heads/tails, spam/not spam). Described by probability p of success.

## In Rust

MachinDeOuf provides all these in `machin-math`:

```rust
use ndarray::array;
use machin_math::stats;

// Basic statistics
let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

let avg = stats::mean(&data).unwrap();             // 5.0
let var = stats::variance(&data).unwrap();          // 4.0 (population)
let svar = stats::sample_variance(&data).unwrap();  // 4.571 (Bessel's)
let std = stats::std_dev(&data).unwrap();           // 2.0
let med = stats::median(&data).unwrap();            // 4.5
let (lo, hi) = stats::min_max(&data).unwrap();     // (2.0, 9.0)

// Covariance and correlation matrices
// Rows = observations, columns = variables
let dataset = array![
    [1.0, 2.0],
    [2.0, 4.0],
    [3.0, 6.0],
    [4.0, 8.0],
];

let cov = stats::covariance_matrix(&dataset).unwrap();
// Strong positive covariance — the two variables move together

let cor = stats::correlation_matrix(&dataset).unwrap();
// cor[0][1] ≈ 1.0 — perfect positive correlation (y = 2x)
```

### Standardization

Many ML algorithms work better when features are on the same scale. Standardization transforms each feature to have mean 0 and standard deviation 1:

```rust
use machin_math::linalg;

let data = array![
    [100.0, 0.1],   // Feature 1 is large, feature 2 is tiny
    [200.0, 0.2],
    [300.0, 0.3],
];

let (standardized, means, stds) = linalg::standardize(&data);
// Now both features have mean ≈ 0, std ≈ 1
// Algorithms like KNN and SVM need this to work properly
```

## When To Use This

| Situation | What You Need |
|-----------|---------------|
| Summarize a dataset | Mean, variance, std dev, median |
| Check if features are related | Covariance/correlation matrix |
| Normalize features before training | Standardize (zero-mean, unit-variance) |
| Build a probabilistic classifier | Bayes' theorem (Naive Bayes) |
| Evaluate model uncertainty | Probability distributions, confidence intervals |
| Compare two models' performance | Statistical tests (not yet in MachinDeOuf) |

## Key Parameters

| Statistic | What It Tells You | Watch Out For |
|-----------|-------------------|---------------|
| Mean | Center of data | Sensitive to outliers |
| Median | Middle value | Robust to outliers, but ignores spread |
| Variance | Spread of data | Squared units (use std dev for interpretation) |
| Std dev | Typical distance from mean | Assumes symmetric spread |
| Correlation | Linear relationship strength | Does NOT mean causation; misses nonlinear patterns |

## Pitfalls

- **Correlation ≠ causation.** Two variables can be perfectly correlated without one causing the other. Ice cream sales and drowning deaths both increase in summer.
- **Mean vs. median.** If your data has outliers (one house costs $50M in a neighborhood of $300K homes), the median is a better summary than the mean.
- **Population vs. sample variance.** Use `sample_variance` (n-1 denominator) when working with a sample from a larger population. Use `variance` (n denominator) when you have the entire population.
- **Standardize before distance-based algorithms.** KNN, SVM, and K-Means all measure distances between points. If one feature ranges 0-1000 and another 0-1, the first feature dominates. Standardize first.

## Going Further

- **Next**: [Calculus Intuition](calculus-intuition.md) — how gradients guide optimization
- **Uses this**: [Naive Bayes](../supervised-learning/naive-bayes.md) uses Bayes' theorem directly
- **Uses this**: [PCA](../unsupervised-learning/pca.md) uses covariance matrices to find principal components
