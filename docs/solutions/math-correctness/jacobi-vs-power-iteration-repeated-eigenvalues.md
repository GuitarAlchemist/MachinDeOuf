---
title: "Jacobi vs power iteration for matrices with repeated eigenvalues"
category: math-correctness
date: 2026-04-09
tags: [eigenvalue, jacobi, power-iteration, linear-algebra, deflation]
symptom: "Classical MDS of a unit square returned pairwise-distance error 1.14 instead of 1e-6; LDA with axis-aligned class clusters produced duplicated or wrong discriminants"
root_cause: "Power iteration + rank-1 deflation converges to an arbitrary vector in the dominant eigenspace when eigenvalues are repeated. Deflation then only removes one direction of a multi-dimensional eigenspace, leaving residual that poisons subsequent iterations."
---

# Jacobi vs power iteration for repeated eigenvalues

## Problem

Implemented classical MDS using power iteration + deflation to extract
the top-k eigenvectors of the double-centered distance matrix. A unit
square test case:

```rust
let dists = array![
    [0.0, 1.0, 1.414, 1.0],
    [1.0, 0.0, 1.0, 1.414],
    [1.414, 1.0, 0.0, 1.0],
    [1.0, 1.414, 1.0, 0.0],
];
let embedding = classical_mds(&dists, 2)?;
```

Expected pairwise-distance error < 1e-6. Got 1.14 — embedding was wildly
wrong. The same problem surfaced later in LDA with three axis-aligned
class clusters in 3D: repeated eigenvalue 1.0 with multiplicity 2 broke
the deflation scheme.

## Root cause

The double-centered distance matrix of a unit square is:

```
 0.5  0  -0.5  0
 0   0.5  0  -0.5
-0.5  0   0.5  0
 0  -0.5  0   0.5
```

Its eigenvalues are `{1, 1, 0, 0}`. Power iteration converges to the
largest-magnitude eigenvalue direction, but for a **2-dimensional**
eigenspace with multiplicity 2, power iteration converges to *some*
arbitrary unit vector in that eigenspace — not necessarily a basis
direction.

Deflation `B' = B - lambda * v * v^T` is rank-1: it removes *one*
direction from the eigenspace, leaving the other direction still active
at the same eigenvalue. The second power iteration then re-converges to
a nearby direction, producing a degenerate result.

This is a well-known limitation of iterative single-vector eigensolvers
on matrices with spectral degeneracy.

## Working solution

Use a **full symmetric eigendecomposition via cyclic Jacobi rotations**
instead of power iteration + deflation. Jacobi handles repeated
eigenvalues correctly because it zeroes off-diagonal entries one pair
at a time without assuming anything about eigenspace dimensionality.

```rust
// crates/ix-math/src/eigen.rs
pub fn symmetric_eigen(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), MathError> {
    // cyclic Jacobi rotations...
    // returns (values_desc_sorted, eigenvectors_as_columns)
}
```

Algorithm outline:

1. Initialize `V = I` (will accumulate rotations)
2. Sweep over all off-diagonal pairs `(p, q)`
3. For each pair, compute Jacobi rotation angle that zeroes `a[p,q]`
4. Apply rotation to rows/columns `p, q` of `a` and columns `p, q` of `V`
5. Stop when off-diagonal Frobenius norm < tolerance
6. Eigenvalues are the diagonal entries of the converged matrix
7. Eigenvectors are the columns of `V`

After this fix, the unit-square MDS test passed with error < 1e-6 and
the LDA three-class test recovered distinct components.

## Prevention

1. **Never use power iteration + deflation on symmetric matrices**
   without first checking for eigenvalue multiplicity. Use Jacobi,
   divide-and-conquer, or a LAPACK binding instead.

2. **Test eigensolvers on matrices with known repeated eigenvalues**:
   - Identity matrix (all eigenvalues = 1)
   - Double-centered distance matrix of a symmetric shape
   - Block-diagonal matrix with identical blocks

3. **Consolidate to one eigensolver** in `ix-math::eigen`. This session
   found 4 duplicated copies in the workspace, all implementing the
   same Jacobi algorithm. Extract to a single canonical home.

## Related

- crates/ix-math/src/eigen.rs — canonical symmetric eigensolver
- docs/solutions/math-correctness/lda-non-symmetric-deflation.md —
  related bug in LDA using the same flawed approach
- commit 67abe77 — extraction of `ix-math::eigen`
- commit ee2e364 — original MDS commit that surfaced the bug
