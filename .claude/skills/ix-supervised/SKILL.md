---
name: ix-supervised
description: Supervised learning — regression, classification, and evaluation metrics
---

# Supervised Learning

Train and evaluate classification and regression models.

## When to Use
When the user asks to classify data, predict values, train a model, evaluate accuracy/precision/recall, or compare ML algorithms.

## Algorithm Selection
- **Linear Regression** — Continuous target, linear relationship, OLS normal equation
- **Logistic Regression** — Binary/multiclass classification, gradient descent
- **SVM** — Binary classification with margin maximization, hinge loss
- **KNN** — Instance-based, no training phase, works with any distribution
- **Naive Bayes** — Fast classification assuming feature independence, good for text
- **Decision Tree** — Non-linear boundaries, interpretable rules, Gini impurity splits

## Programmatic Usage
```rust
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::svm::LinearSVM;
use ix_supervised::knn::KNN;
use ix_supervised::naive_bayes::GaussianNaiveBayes;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::{Regressor, Classifier};
use ix_supervised::metrics;
```

## MCP Tool Reference
Tool: `ix_supervised` — Operations: `linear_regression`, `logistic_regression`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `metrics`
