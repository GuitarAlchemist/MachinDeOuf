---
date: 2026-04-09
topic: dimensionality-reduction-coverage
---

# Comprehensive Dimensionality Reduction Coverage for ix

## Taxonomy (what the user asked for)

### 1. Feature Selection (keep subset of original variables)

**Filter methods** (statistical ranking, model-independent):
- Correlation coefficients
- Chi-square
- Information gain
- Missing value ratio / variance threshold

**Wrapper methods** (use a predictive model):
- Forward selection
- Backward elimination
- Recursive Feature Elimination (RFE)

**Embedded methods** (during training):
- LASSO (L1 regularization)
- Decision tree importance
- Random forest importance

### 2. Linear Feature Extraction (linear combinations of originals)

- Principal Component Analysis (PCA) -- unsupervised, max variance
- Linear Discriminant Analysis (LDA) -- supervised, max class separation
- Singular Value Decomposition (SVD) -- foundational
- Independent Component Analysis (ICA) -- statistically independent components
- Non-negative Matrix Factorization (NMF) -- non-negative decomposition
- Factor Analysis (FA) -- latent factors plus noise

### 3. Non-linear Feature Extraction / Manifold Learning

- t-SNE -- local neighbor preservation, visualization
- UMAP -- fuzzy simplicial sets, faster than t-SNE
- Isomap -- geodesic distances on manifold
- Kernel PCA -- PCA in RKHS
- Autoencoders -- neural bottleneck
- MDS -- preserve pairwise distances
- LLE -- local linear combinations

## ix Current State (inventory)

| Technique | Status in ix | Location |
|-----------|-------------|----------|
| PCA | ✅ | `ix-unsupervised::pca` |
| t-SNE | ✅ | `ix-unsupervised::tsne` |
| Poincare hierarchy (hyperbolic embedding) | ✅ | `ix-math::poincare_hierarchy` |
| Decision tree (for importance) | ✅ | `ix-supervised::decision_tree` |
| Random forest (for importance) | ✅ | `ix-ensemble` |
| Linear regression | ✅ (no L1 yet) | `ix-supervised::linear_regression` |
| Stats (for correlation) | ✅ | `ix-math::stats` |
| Power iteration / eigendecomposition | ✅ (local to PCA) | `ix-unsupervised::pca` |
| **SVD** | ❌ | -- |
| **LDA** | ❌ | -- |
| **NMF** | ❌ | -- |
| **Factor Analysis** | ❌ | -- |
| **ICA** | ❌ | -- |
| **Kernel PCA** | ❌ | -- |
| **MDS** | ❌ | -- |
| **Isomap** | ❌ | -- |
| **LLE** | ❌ | -- |
| **UMAP** | ❌ | -- |
| **Autoencoder** (dedicated) | ❌ | (can build on ix-nn) |
| **LASSO** | ❌ | -- |
| **RFE** | ❌ | -- |
| **Correlation filter** | ❌ (stats exists) | -- |
| **Variance threshold** | ❌ | -- |
| **Forward/backward selection** | ❌ | -- |

## Implementation Plan (atomic commits, in priority order)

Each commit must: build, test, clippy-clean, no new dependencies outside workspace.

### Round 1: Foundations (no dependencies on other new crates)

1. **SVD** -- ~300 LOC, `ix-math::svd` via one-sided Jacobi or Golub-Kahan bidiagonalization. Enables PCA refactor, LSI, pseudo-inverse.
2. **LDA** -- ~200 LOC, `ix-supervised::lda`. Within/between-class scatter, generalized eigenvalue. Uses ix-math linalg.
3. **NMF** -- ~200 LOC, `ix-unsupervised::nmf`. Multiplicative update rules.
4. **MDS** -- ~150 LOC, `ix-unsupervised::mds`. Classical MDS via double-centered distance matrix eigendecomp.
5. **Kernel PCA** -- ~200 LOC, `ix-unsupervised::kernel_pca`. RBF, polynomial, linear kernels. Apply PCA to Gram matrix.

### Round 2: Feature Selection (small, fast wins)

6. **Variance threshold filter** -- ~50 LOC, `ix-supervised::feature_selection::variance_threshold`.
7. **Correlation filter** -- ~100 LOC, same module. Pairwise correlation with target, rank features.
8. **Mutual information / information gain filter** -- ~150 LOC.
9. **Missing value ratio filter** -- ~50 LOC.
10. **Recursive Feature Elimination (RFE)** -- ~200 LOC, wraps any `Regressor`/`Classifier` trait.
11. **Forward selection / backward elimination** -- ~200 LOC each.

### Round 3: Embedded / Regularization

12. **LASSO regression** -- ~300 LOC, `ix-supervised::lasso`. Coordinate descent with soft-thresholding.
13. **Elastic Net** -- ~200 LOC (extension of LASSO).
14. **Random Forest feature importance helper** -- ~100 LOC, expose existing RF internals.

### Round 4: Manifold Learning (non-linear extraction)

15. **Isomap** -- ~250 LOC, `ix-unsupervised::isomap`. Uses ix-graph kNN + Dijkstra + MDS.
16. **LLE (Locally Linear Embedding)** -- ~300 LOC, `ix-unsupervised::lle`. Local weights + low-dim embedding.
17. **ICA (FastICA)** -- ~400 LOC, `ix-unsupervised::ica`. Whitening + fixed-point iteration.
18. **Factor Analysis (EM)** -- ~350 LOC, `ix-unsupervised::factor_analysis`.
19. **UMAP** -- ~700-1000 LOC, `ix-unsupervised::umap`. Biggest; fuzzy simplicial sets + SGD.

### Round 5: Neural Approaches

20. **Autoencoder** -- ~300 LOC, `ix-nn::autoencoder`. Wraps existing dense layers with encoder/decoder split.

## Total Scope

~4000-5000 LOC across ~20 atomic commits. Every commit independently
tested, documented, and clippy-clean.

## Priority for this session

Start with SVD (foundational) and work through Round 1 as atomic commits.
Later rounds can happen in follow-up sessions.
