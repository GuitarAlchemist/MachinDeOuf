# Vectors & Matrices

> The fundamental data structures of machine learning — what they are, why they matter, and how to use them in Rust.

## The Problem

You have a dataset of houses. Each house has a size (sq ft), number of bedrooms, age (years), and a price. You want a computer to learn the relationship between house features and price.

But computers don't understand "houses." They understand numbers. Specifically, they understand *lists of numbers* and *tables of numbers*. That's exactly what vectors and matrices are.

## The Intuition

### Vectors: A List of Numbers

A **vector** is just an ordered list of numbers. That's it.

A single house with 3 features becomes a vector:
```
house = [1500, 3, 10]
         ↑     ↑   ↑
       sq ft  beds  age
```

Think of a vector as an arrow in space. A 2D vector `[3, 4]` is an arrow pointing 3 units right and 4 units up. A 3D vector adds depth. An ML vector with 100 features is an arrow in 100-dimensional space — impossible to visualize, but the math works identically.

**Key operations you'll see everywhere:**

- **Addition**: `[1, 2] + [3, 4] = [4, 6]` — combine two vectors element by element
- **Scaling**: `2 * [1, 2] = [2, 4]` — stretch or shrink a vector
- **Dot product**: `[1, 2] · [3, 4] = 1×3 + 2×4 = 11` — measures how much two vectors point in the same direction. This is the single most important operation in ML.
- **Norm (length)**: `||[3, 4]|| = √(9+16) = 5` — the magnitude of the vector

### Matrices: A Table of Numbers

A **matrix** is a rectangular grid of numbers. Think of it as multiple vectors stacked together.

Your dataset of 4 houses becomes a matrix:
```
X = | 1500  3  10 |    ← house 1
    | 2000  4   5 |    ← house 2
    | 1200  2  20 |    ← house 3
    | 1800  3   8 |    ← house 4
```

This is a 4×3 matrix (4 rows, 3 columns). Each row is a data point. Each column is a feature.

**Why matrices matter:** Almost every ML algorithm boils down to matrix operations. Linear regression? Matrix multiplication. Neural networks? Chains of matrix multiplications with nonlinear functions in between. PCA? Finding special vectors of a matrix.

### Matrix Multiplication

The most important operation. When you multiply a matrix by a vector, you get a new vector. In plain English, matrix multiplication *transforms* data.

```
| 2  0 |     | 3 |     | 6 |
| 0  3 |  ×  | 2 |  =  | 6 |
```

In plain English: the matrix `[[2,0],[0,3]]` scales x by 2 and y by 3. Matrix multiplication is how models make predictions — the matrix holds the learned weights, and the vector is your input data.

## How It Works

### Vector Operations in Detail

**Dot product** (also called inner product):

Given two vectors a = [a₁, a₂, ..., aₙ] and b = [b₁, b₂, ..., bₙ]:

`a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`

In plain English, this multiplies matching elements and sums them up. The result is a single number.

Why it matters: The dot product tells you how similar two vectors are. If they point the same direction, the dot product is large and positive. If perpendicular, it's zero. If opposite, it's large and negative.

**Euclidean norm** (length of a vector):

`||a|| = √(a₁² + a₂² + ... + aₙ²)`

In plain English, this is the Pythagorean theorem generalized to any number of dimensions.

### Matrix Operations in Detail

**Matrix-vector multiplication** (transforms a vector):

For a 2×2 matrix M and vector v:

```
| m₁₁  m₁₂ |     | v₁ |     | m₁₁v₁ + m₁₂v₂ |
| m₂₁  m₂₂ |  ×  | v₂ |  =  | m₂₁v₁ + m₂₂v₂ |
```

Each row of the result is the dot product of a matrix row with the vector.

**Matrix-matrix multiplication**:

To multiply A (m×k) by B (k×n), the inner dimensions must match. The result is m×n. Each element (i,j) of the result is the dot product of row i of A with column j of B.

**Transpose** (flip rows and columns):

```
| 1  2  3 |  transpose   | 1  4 |
| 4  5  6 |  ────────→   | 2  5 |
                          | 3  6 |
```

**Determinant** (scalar that describes a square matrix):

For a 2×2 matrix: `det([[a,b],[c,d]]) = ad - bc`

In plain English: the determinant tells you how much the matrix scales areas. If it's zero, the matrix squishes everything onto a lower dimension (the matrix is "singular" and can't be inverted).

**Inverse** (the "undo" matrix):

If M × M⁻¹ = I (identity matrix), then M⁻¹ is the inverse of M. Not every matrix has an inverse — only square matrices with nonzero determinant.

## In Rust

MachinDeOuf uses `ndarray` for all vector and matrix operations. The `machin-math` crate adds higher-level functions.

```rust
use ndarray::{array, Array1, Array2};
use machin_math::linalg;

// Vectors
let a: Array1<f64> = array![1.0, 2.0, 3.0];
let b: Array1<f64> = array![4.0, 5.0, 6.0];

// Dot product
let dot = a.dot(&b);  // 1*4 + 2*5 + 3*6 = 32.0

// Norm (length)
let norm = a.dot(&a).sqrt();  // √(1+4+9) = √14

// Element-wise operations
let sum = &a + &b;     // [5.0, 7.0, 9.0]
let scaled = &a * 2.0; // [2.0, 4.0, 6.0]

// Matrices
let m = array![[1.0, 2.0], [3.0, 4.0]];
let v = array![1.0, 1.0];

// Matrix-vector multiply
let result = linalg::matvec(&m, &v).unwrap();  // [3.0, 7.0]

// Matrix-matrix multiply
let a_mat = array![[1.0, 2.0], [3.0, 4.0]];
let b_mat = array![[5.0, 6.0], [7.0, 8.0]];
let product = linalg::matmul(&a_mat, &b_mat).unwrap();

// Transpose, determinant, inverse
let t = linalg::transpose(&m);
let det = linalg::determinant(&m).unwrap();      // 1*4 - 2*3 = -2.0
let inv = linalg::inverse(&m).unwrap();

// Identity matrix
let eye = linalg::eye(3);  // 3×3 identity

// Column means and standardization
let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
let means = linalg::col_mean(&data);                          // [2.0, 20.0]
let (standardized, means, stds) = linalg::standardize(&data); // zero-mean, unit-variance
```

## When To Use This

You don't choose to use vectors and matrices — they're the default representation for everything in ML:

| Data Type | Representation |
|-----------|---------------|
| A single data point | Vector (`Array1`) |
| A dataset | Matrix (`Array2`) — rows are samples, columns are features |
| Model weights | Vector or matrix |
| Predictions | Vector |
| Transformation (rotation, scaling) | Matrix |

## Key Parameters

| Operation | Dimension Rule | Fails When |
|-----------|---------------|------------|
| `a + b` (vectors) | Same length | Different lengths |
| `a · b` (dot product) | Same length | Different lengths |
| `M × v` (matrix-vector) | M is m×n, v has n elements | Column count ≠ vector length |
| `A × B` (matrix-matrix) | A is m×k, B is k×n | A's columns ≠ B's rows |
| `det(M)` | M must be square | Non-square matrix |
| `M⁻¹` | M must be square | Singular matrix (det = 0) |

## Pitfalls

- **Dimension mismatches** are the #1 source of errors. Always check your array shapes with `.dim()` or `.shape()`.
- **Row vs. column conventions**: In MachinDeOuf, datasets are always rows=samples, columns=features. Some textbooks use the transpose convention.
- **Singular matrices**: If you get a "matrix is singular" error, your data probably has linearly dependent features (one feature is a multiple of another). Try removing redundant features or using regularization.
- **Numerical precision**: `f64` gives ~15 decimal digits. For most ML, this is plenty. But if you're inverting large matrices, small errors can accumulate — prefer algorithms that avoid explicit matrix inversion when possible.

## Going Further

- **Next**: [Probability & Statistics](probability-and-statistics.md) — the mathematical language of uncertainty
- **See also**: [Rust for ML](rust-for-ml.md) for `ndarray` patterns
- **Deeper**: [PCA](../unsupervised-learning/pca.md) uses eigendecomposition of the covariance matrix
