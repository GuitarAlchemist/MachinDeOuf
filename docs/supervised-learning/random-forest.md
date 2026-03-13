# Random Forest

## The Problem

A credit card company monitors millions of transactions daily. Each transaction has features: the amount, the merchant category, the time of day, the distance from the cardholder's home, whether the card was physically present, and dozens more. Fraudulent transactions are rare -- roughly 1 in 1,000 -- but each one costs real money. The company needs a model that catches fraud reliably without blocking legitimate purchases.

A single decision tree can learn patterns like "transactions over $5,000 at electronics stores at 3 AM are suspicious," but it's brittle. Change a few training examples and the tree looks completely different. It also tends to overfit -- memorizing the specific fraud cases it trained on rather than learning general patterns.

What if you built 100 different decision trees, each looking at a slightly different random sample of the data and a slightly different random subset of features, then let them vote? The errors of individual trees would cancel out, the ensemble would be far more robust, and you'd get better fraud detection. That's a random forest.

## The Intuition

Imagine asking 100 different loan officers to review the same application, but each officer only sees a random subset of the paperwork and a random subset of past cases. Some will make mistakes, but their mistakes will be *different* mistakes. When you tally the votes, the majority answer is almost always right, because it's unlikely that most officers made the same wrong call.

This is the "wisdom of crowds" principle applied to decision trees. Each tree is deliberately made a little imperfect (by training on a random bootstrap sample with a random subset of features), and the collective averaging smooths out individual errors.

The two sources of randomness are:
1. **Bagging (bootstrap aggregating):** Each tree trains on a random sample *with replacement* from the training data. Some examples appear multiple times; others are left out entirely.
2. **Feature subsampling:** At each split, the tree only considers a random subset of features (typically the square root of the total number of features). This decorrelates the trees -- if one feature is very strong, not every tree will use it at the root.

## How It Works

### Step 1: Create bootstrap samples

For each tree, draw n samples *with replacement* from the original n training examples.

In plain English, this means: imagine putting all your training examples in a bag, drawing one out (and putting it back), and repeating n times. Some examples get drawn multiple times, and roughly 37% never get drawn at all. Each tree sees a different "version" of the training data.

### Step 2: Select a random feature subset

At each split in each tree, randomly choose `max_features` features (default: sqrt(total features)) and only consider splits on those features.

$$
m = \lceil \sqrt{p} \rceil
$$

where p is the total number of features.

In plain English, this means: if you have 16 features, each split only looks at 4 randomly chosen ones. This prevents every tree from using the same dominant feature at the root, which would make all trees correlated and defeat the purpose of the ensemble.

### Step 3: Grow each tree

Using the bootstrap sample and the feature subsampling rule, grow a decision tree using the CART algorithm with Gini impurity (see [decision-trees.md](./decision-trees.md)).

In plain English, this means: each tree asks its questions using only its assigned features and its bootstrapped data. The trees are grown to the specified `max_depth`.

### Step 4: Aggregate predictions (voting)

For classification, each tree votes for a class. The forest's prediction is the majority vote. For probability estimates, the forest averages the class probability distributions across all trees.

$$
P(\text{class} = c \mid x) = \frac{1}{T} \sum_{t=1}^{T} P_t(\text{class} = c \mid x)
$$

In plain English, this means: ask every tree "is this transaction fraud?" and go with whatever answer most trees agree on. For probabilities, average how confident each tree is.

## In Rust

```rust
use ndarray::array;
use machin_ensemble::random_forest::RandomForest;
use machin_ensemble::traits::EnsembleClassifier;
use machin_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Features: [amount, merchant_cat, hour, distance_km, card_present]
    let x = array![
        [25.0,  1.0,  12.0,  2.0,  1.0],   // legitimate
        [15.0,  2.0,  14.0,  1.0,  1.0],   // legitimate
        [120.0, 3.0,  10.0,  5.0,  1.0],   // legitimate
        [45.0,  1.0,  18.0,  0.5,  1.0],   // legitimate
        [4500.0, 5.0,  3.0, 800.0, 0.0],   // fraud
        [2200.0, 5.0,  2.0, 500.0, 0.0],   // fraud
        [3100.0, 4.0,  4.0, 650.0, 0.0],   // fraud
        [5000.0, 5.0,  1.0, 900.0, 0.0],   // fraud
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1]; // 0 = legitimate, 1 = fraud

    // Build a random forest: 50 trees, max depth 5, seeded for reproducibility
    let mut forest = RandomForest::new(50, 5)
        .with_seed(42)
        .with_max_features(3);  // consider 3 of 5 features per split
    forest.fit(&x, &y);

    println!("Number of trees: {}", forest.n_estimators());

    // Predict
    let predictions = forest.predict(&x);
    println!("Predictions: {}", predictions);

    // Probability estimates
    let proba = forest.predict_proba(&x);
    println!("Fraud probabilities:");
    for i in 0..proba.nrows() {
        println!("  Transaction {}: {:.1}% fraud", i, proba[[i, 1]] * 100.0);
    }

    // Evaluate -- for fraud detection, precision and recall matter more than accuracy
    println!("Accuracy:  {:.4}", accuracy(&y, &predictions));
    println!("Precision (fraud): {:.4}", precision(&y, &predictions, 1));
    println!("Recall (fraud):    {:.4}", recall(&y, &predictions, 1));
    println!("F1 (fraud):        {:.4}", f1_score(&y, &predictions, 1));
}
```

## When To Use This

| Situation | Random Forest | Alternative | Why |
|---|---|---|---|
| Tabular data, need high accuracy | Yes | -- | Consistently top performer on structured data |
| Want low-maintenance model | Yes | -- | Works well out of the box with minimal tuning |
| Need probability calibration | Decent | Logistic regression | Forest probabilities are reasonable but not perfectly calibrated |
| Need single interpretable model | No | Decision tree | A forest of 50 trees is a black box |
| Feature importance needed | Yes | -- | Track which features are used most across all trees |
| Very high-dimensional data (1000+ features) | Yes | -- | Feature subsampling handles this naturally |
| Real-time latency-critical prediction | Maybe | Logistic regression | Predicting through 50 trees is slower than one dot product |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_trees` | (required) | Number of trees in the forest. More trees = better accuracy, slower training. |
| `max_depth` | (required) | Maximum depth of each individual tree. |
| `max_features` | `sqrt(n_features)` | Number of features considered at each split. |
| `seed` | `42` | Random seed for reproducibility (bootstrap sampling and feature selection). |

Configure via the builder pattern:
```rust
RandomForest::new(100, 10)      // 100 trees, max depth 10
    .with_seed(123)
    .with_max_features(4)
```

### How many trees?

| n_trees | Behavior |
|---|---|
| 1 | Just a single tree -- no ensemble benefit |
| 10-50 | Good for quick experiments |
| 100-500 | Standard production range |
| 1000+ | Diminishing returns, much slower training |

Error typically drops steeply with the first 20-30 trees, then flattens. You can plot out-of-bag error vs. n_trees to find the sweet spot.

## Pitfalls

**Not interpretable.** Unlike a single decision tree, you cannot trace a random forest's reasoning as a simple flowchart. If regulatory explainability is mandatory, use a single decision tree or logistic regression instead.

**Slow on large datasets.** Training 100 trees on a million rows with 50 features takes time. Each tree is independent, so this is embarrassingly parallelizable -- but the current MachinDeOuf implementation is single-threaded.

**Overfitting with deep trees.** While forests are more resistant to overfitting than single trees, setting `max_depth` too high on small datasets can still cause problems. Start with `max_depth = 10` and tune from there.

**Class imbalance.** In fraud detection, fraudulent transactions may be 0.1% of the data. The forest may learn to always predict "legitimate" because that's right 99.9% of the time. Use stratified sampling, class weights, or evaluation metrics that account for imbalance (see [evaluation-metrics.md](./evaluation-metrics.md)).

**Correlated trees.** If one feature is overwhelmingly dominant, even with feature subsampling many trees may end up using it at the root. Reducing `max_features` further helps decorrelate the trees.

## Going Further

- **Out-of-bag (OOB) estimation:** The ~37% of samples not used to train each tree can serve as a built-in validation set. This lets you estimate generalization error without a separate test split.
- **Feature importance:** Count how often each feature is used across all splits, weighted by the impurity decrease. Features that appear near the root of many trees are the most important.
- **Gradient boosting:** Instead of training trees in parallel on bootstrapped data, train them sequentially where each tree corrects the errors of the previous one. The `machin-ensemble` crate has a boosting stub for future development.
- **Individual tree analysis:** Since each tree is a `DecisionTree` from `machin-supervised`, you can inspect individual trees for debugging.
