---
name: ix-random-forest
description: Random forest classifier — train and predict with ensemble of decision trees
---

# Random Forest

Ensemble classifier using bootstrap aggregation of decision trees.

## When to Use
When the user needs classification with probability estimates, wants a robust baseline classifier, or has tabular data.

## Capabilities
- **Train** — Fit a random forest on labeled data
- **Predict** — Majority-vote class predictions
- **Predict probabilities** — Per-class probability estimates
- **Configurable** — Number of trees, max depth, random seed

## Key Concepts
- Each tree trained on bootstrap sample (bagging)
- Random feature subset at each split reduces correlation
- More trees → better performance (diminishing returns)

## Programmatic Usage
```rust
use ix_ensemble::random_forest::RandomForest;
use ix_ensemble::traits::EnsembleClassifier;
```

## MCP Tool
Tool name: `ix_random_forest`
Parameters: `x_train`, `y_train`, `x_test`, `n_trees`, `max_depth`
