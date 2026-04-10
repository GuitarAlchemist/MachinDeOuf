---
title: "LDA non-symmetric deflation produces duplicated or wrong discriminants"
category: math-correctness
date: 2026-04-09
tags: [lda, eigenvalue, generalized-eigenvalue, deflation, symmetrization]
symptom: "LDA with three axis-aligned class clusters returned duplicated components; the second explained value collapsed to near zero"
root_cause: "Naive LDA formulation forms M = S_W^-1 S_B which is NOT symmetric, then uses rank-1 symmetric deflation M -= lambda v v^T which is only valid for Hermitian matrices"
---

# LDA non-symmetric deflation bug

## Problem

The classical Fisher LDA objective is the generalized eigenvalue problem
`S_B v = lambda S_W v`, typically solved by forming
`M = S_W^{-1} S_B` and applying power iteration + deflation:

```rust
// BAD: deflation assumes M is symmetric, but S_W^-1 S_B is NOT
let m = sw_inv.dot(&sb);
for k in 0..n_components {
    let (lambda, v) = power_iteration(&current, 500, 1e-10);
    components.row_mut(k).assign(&v);
    // This deflation is only valid for symmetric M
    current -= lambda * outer(&v, &v);
}
```

Regression symptoms:
- Three classes arranged along the three coordinate axes in 3D feature
  space produced LDA components that separated only two of the three
  classes.
- The `explained` array had one non-trivial value and one near-zero
  value, even though the problem has two genuine discriminant
  directions of equal importance.
- The error was silent — no panics, no warnings, just wrong math.

## Root cause

`S_W^{-1} S_B` is in general **not symmetric**: `A^T B != B A^T` unless
the two factors commute, which only happens when they share an
eigenbasis (uncommon).

Rank-1 deflation `M -= lambda v v^T` is only correct for symmetric
(Hermitian) matrices because it assumes `v` is both a left and right
eigenvector. For non-symmetric matrices you need *both* the left and
right eigenvectors, and you deflate with `M -= lambda * u * v^T` where
`u` is the matched left eigenvector. The code had neither check nor
the matched pair.

The axis-aligned test case exposes this because `S_W^{-1} S_B` has
repeated eigenvalues (symmetry of the class layout), and power
iteration in the 2D eigenspace returns an arbitrary direction.
Deflation then only removes one direction, leaving the eigenspace
active for the next iteration — which converges to a nearly parallel
direction.

## Working solution

**Symmetrize the generalized eigenvalue problem.** Use the inverse
square root of `S_W`:

```text
  Let S_W = U diag(d) U^T                    (symmetric eigendecomp)
  Let S_W^{-1/2} = U diag(1/sqrt(d)) U^T
  Define M = S_W^{-1/2} S_B S_W^{-1/2}       (SYMMETRIC by construction)
  Solve M u_k = lambda_k u_k                  (standard sym eigendecomp)
  LDA directions: v_k = S_W^{-1/2} u_k
```

Implementation uses the canonical `ix-math::eigen::symmetric_eigen`
twice: once for `S_W`, once for the symmetrized `M`.

```rust
use ix_math::eigen::symmetric_eigen;

// Step 1: eigendecompose S_W
let (sw_vals, sw_vecs) = symmetric_eigen(&sw)?;

// Step 2: S_W^{-1/2} = U diag(1/sqrt(d)) U^T
let mut sw_inv_sqrt = Array2::zeros((n_features, n_features));
for k in 0..n_features {
    let inv_sqrt_d = 1.0 / sw_vals[k].sqrt();
    for i in 0..n_features {
        for j in 0..n_features {
            sw_inv_sqrt[[i, j]] +=
                inv_sqrt_d * sw_vecs[[i, k]] * sw_vecs[[j, k]];
        }
    }
}

// Step 3: symmetric M
let m = sw_inv_sqrt.dot(&sb).dot(&sw_inv_sqrt);

// Step 4: eigendecompose M (symmetric — handles repeated eigenvalues)
let (m_vals, m_vecs) = symmetric_eigen(&m)?;

// Step 5: map back
for k in 0..n_components {
    let u_k = m_vecs.column(k);
    let v_k = sw_inv_sqrt.dot(&u_k);  // this is the LDA direction
    components.row_mut(k).assign(&v_k);
}
```

## Prevention

1. **Never form `A^{-1} B` and treat it as symmetric** — it almost
   never is. The correct generalized-eigenvalue approach is always
   symmetrization via `A^{-1/2}`.

2. **Require positive-definiteness on `S_W`** and return an error if
   any eigenvalue is non-positive. A tiny diagonal regularization
   (1e-10) keeps well-behaved inputs invertible, but if it's not enough,
   the caller has a data-quality problem we should surface.

3. **Add a regression test** with axis-aligned class clusters. This is
   the minimal example that exposes the bug — if a new LDA variant
   (e.g. kernel LDA, regularized LDA) passes this test, the eigenvalue
   extraction is probably correct.

4. **Document the math** in the module docs. The symmetrization step is
   not obvious from reading the canonical Fisher paper; future readers
   will benefit from seeing `M = S_W^{-1/2} S_B S_W^{-1/2}` spelled out.

## Related

- crates/ix-unsupervised/src/lda.rs — fixed implementation
- docs/solutions/math-correctness/jacobi-vs-power-iteration-repeated-eigenvalues.md
  — broader context on the eigenvalue degeneracy issue
- commit 789cf54 — the fix
- Multi-AI review finding #1 (Codex GPT-5.4)
