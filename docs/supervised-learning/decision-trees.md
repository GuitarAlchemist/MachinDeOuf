# Decision Trees

## The Problem

A bank receives thousands of loan applications per day. Each application contains structured data: the applicant's annual income, credit score, employment length, existing debt, and the requested loan amount. A human loan officer looks at these factors and decides to approve or deny. The bank wants to automate this process, but there's a catch -- regulators require that every denial can be explained. "The algorithm said no" is not an acceptable answer. They need a model whose reasoning can be traced step by step.

Decision trees solve this problem because they make decisions exactly the way a human would describe them: "If income is above $60K *and* credit score is above 700 *and* debt-to-income ratio is below 0.4, approve the loan. Otherwise, deny." The model is a flowchart. You can print it, hand it to a regulator, and they can follow the logic from root to leaf.

Unlike linear models that produce a single weighted sum, a decision tree partitions the feature space into rectangular regions, each assigned a class label. This lets it capture non-linear patterns and interactions between features without any feature engineering.

## The Intuition

Imagine you're organizing a messy drawer of documents into labeled folders. You pick up the whole stack and ask: "What single question splits this stack into the most homogeneous piles?" Maybe it's "Is the income above $50K?" You split the stack. Now you pick up the left pile and ask another question. And another. You keep splitting until each pile contains only loan approvals or only denials.

That's a decision tree. Each question is a "split" on a feature at a threshold. Each final pile is a "leaf" with a predicted class. The art is choosing the *right* questions -- the ones that create the purest piles fastest.

The algorithm measures pile purity using Gini impurity: a pile that is 100% approvals has Gini = 0 (pure). A pile that is 50/50 has Gini = 0.5 (maximally impure). At each step, the tree picks the question that reduces the average Gini impurity the most.

## How It Works

### Step 1: Measure impurity with the Gini index

$$
\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2
$$

where p_c is the fraction of samples in set S that belong to class c.

In plain English, this means: if you randomly pick two samples from the pile and they're very likely to be the same class, the pile is pure (low Gini). If it's a coin flip, the pile is impure (high Gini).

### Step 2: Evaluate every possible split

For each feature and each possible threshold (midpoint between consecutive distinct values):

$$
\text{Gain} = \text{Gini}(\text{parent}) - \frac{n_L}{n} \text{Gini}(L) - \frac{n_R}{n} \text{Gini}(R)
$$

In plain English, this means: try every way to split the data, measure how much purer the two resulting piles are compared to the original, and pick the split with the biggest improvement.

### Step 3: Recurse

Apply the best split to create two child nodes. Repeat the process on each child until one of the stopping conditions is met:
- Maximum tree depth reached
- Fewer than `min_samples_split` samples in the node
- The node is already pure (Gini = 0)

In plain English, this means: keep asking questions until the piles are pure enough, or until you've asked enough questions (depth limit). The depth limit prevents the tree from memorizing noise.

### Step 4: Predict

For a new sample, walk down the tree following the splits. The leaf node you arrive at gives you the predicted class (the majority class in that leaf) and class probabilities (the distribution of training samples in that leaf).

In plain English, this means: answer each question in the flowchart, follow the arrow, and read the answer at the bottom.

## In Rust

```rust
use ndarray::array;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, precision, recall};

fn main() {
    // Features: [annual_income_k, credit_score, employment_years, debt_to_income]
    let x = array![
        [85.0, 750.0, 10.0, 0.20],   // approved
        [45.0, 620.0,  2.0, 0.55],   // denied
        [70.0, 710.0,  5.0, 0.30],   // approved
        [30.0, 580.0,  1.0, 0.70],   // denied
        [95.0, 780.0, 15.0, 0.15],   // approved
        [50.0, 640.0,  3.0, 0.50],   // denied
        [60.0, 690.0,  4.0, 0.35],   // approved
        [35.0, 590.0,  1.0, 0.65],   // denied
    ];
    let y = array![1, 0, 1, 0, 1, 0, 1, 0]; // 1 = approved, 0 = denied

    // Build a tree with max depth 3, requiring at least 2 samples to split
    let mut tree = DecisionTree::new(3).with_min_samples_split(2);
    tree.fit(&x, &y);

    // Predict on training data
    let predictions = tree.predict(&x);
    println!("Predictions: {}", predictions);
    println!("Accuracy: {:.2}%", accuracy(&y, &predictions) * 100.0);

    // Get probability estimates for a new applicant
    let new_applicant = array![[65.0, 700.0, 6.0, 0.32]];
    let proba = tree.predict_proba(&new_applicant);
    println!(
        "Denial probability: {:.1}%, Approval probability: {:.1}%",
        proba[[0, 0]] * 100.0,
        proba[[0, 1]] * 100.0
    );

    // Evaluate
    println!("Precision (approved): {:.4}", precision(&y, &predictions, 1));
    println!("Recall (approved):    {:.4}", recall(&y, &predictions, 1));
}
```

## When To Use This

| Situation | Decision Tree | Alternative | Why |
|---|---|---|---|
| Explainability required | Yes | -- | Decisions can be traced as if/else rules |
| Mixed feature types, non-linear patterns | Yes | -- | Handles interactions without feature engineering |
| High accuracy on complex data | No | Random forest, gradient boosting | Single trees overfit or underfit |
| Smooth decision boundaries needed | No | Logistic regression, SVM | Trees produce axis-aligned rectangular boundaries |
| Handling missing data (future work) | Maybe | -- | CART can be extended to handle missing values |
| Small dataset | Yes | -- | Trees are quick to train and easy to validate |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | (required) | Maximum depth of the tree. Controls overfitting. |
| `min_samples_split` | `2` | Minimum number of samples required to attempt a split. Higher values create simpler trees. |

Configure via the builder pattern:
```rust
DecisionTree::new(5)              // max_depth = 5
    .with_min_samples_split(10)   // need at least 10 samples to split
```

### Depth vs. overfitting

| max_depth | Behavior |
|---|---|
| 1 | "Decision stump" -- one question, very simple, likely underfits |
| 3-5 | Good starting range for most problems |
| 10-20 | Very detailed, risk of memorizing training noise |
| Unlimited (large value) | Will perfectly fit training data, usually terrible on new data |

## Pitfalls

**Overfitting.** A deep tree will memorize training data, including noise. Always set `max_depth` to a reasonable value and validate on held-out data. A tree that gets 100% training accuracy and 60% test accuracy is overfitting.

**Instability.** Small changes in the training data can produce completely different trees. One removed data point might change the root split, cascading through the entire structure. This is a fundamental property of trees -- random forests address it by averaging many trees.

**Axis-aligned splits only.** The tree can only split on one feature at a time with a threshold (e.g., "income > 50K"). It cannot natively represent diagonal decision boundaries like "income + 0.5 * credit_score > 400". It approximates diagonals with a staircase of axis-aligned splits, which can require a very deep tree.

**Biased toward features with many distinct values.** A continuous feature with 1000 unique values offers more possible split points than a binary feature, so the tree may favor it even when the binary feature is more predictive. This is a known property of CART.

**No extrapolation.** Like all tree-based methods, predictions on data outside the training range will simply return the value of the nearest leaf. A tree trained on incomes from $30K to $100K will predict the same approval probability for $200K as it does for $100K.

## Going Further

- **Random forests:** Combine many decision trees to reduce variance and improve accuracy. See [random-forest.md](./random-forest.md).
- **Feature importance:** Features used in splits near the root are more important. Track which features appear at which depths to understand what drives decisions.
- **Pruning:** Train a deep tree, then remove branches that don't improve validation accuracy. This is an alternative to setting `max_depth` upfront.
- **Multi-class:** The ix implementation handles any number of classes -- labels are `Array1<usize>` with classes 0, 1, 2, etc.
