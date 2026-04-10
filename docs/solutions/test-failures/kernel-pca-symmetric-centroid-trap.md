---
title: "Kernel PCA test assertion fails on symmetric input — centroids collapse to zero"
category: test-failures
date: 2026-04-09
tags: [kernel-pca, test-design, symmetry, centroid]
symptom: "Test asserting that Kernel PCA separates 'inner ring from outer ring' failed because both rings had mean projection ~0"
root_cause: "Both rings were centered at the origin. PCA projections of symmetric point clouds have centroids at the origin in ALL components, regardless of which kernel is used"
---

# Kernel PCA centroid test trap

## Problem

Wrote a Kernel PCA test intended to verify that RBF kernel separates
two concentric rings:

```rust
#[test]
fn test_rbf_kernel_separates_rings() {
    let x = array![
        // inner ring (4 points at radius 1)
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        // outer ring (4 points at radius 2)
        [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
    ];
    let mut kpca = KernelPca::new(2, Kernel::Rbf { gamma: 0.5 });
    let projected = kpca.fit_transform(&x)?;

    // BAD: assumes centroids differ
    let inner_mean = projected.slice(s![0..4, 0]).mean().unwrap();
    let outer_mean = projected.slice(s![4..8, 0]).mean().unwrap();
    assert!((inner_mean - outer_mean).abs() > 0.01);  // FAILS
}
```

The test failed because `inner_mean` and `outer_mean` were both
essentially zero.

## Root cause

Both rings are perfectly symmetric about the origin: for every point
`(x, y)` in a ring, the point `(-x, -y)` is also in the ring. After
Kernel PCA's centering (which subtracts row/column means and adds back
the grand mean), the projection preserves this symmetry — every
component of the projection has zero mean within each symmetric ring.

This is not a Kernel PCA bug. It's a test-design mistake: asserting
that "centroids differ" for two point clouds that are *both* centered
at the origin is always false, regardless of whether the transformation
preserved structure.

## Working solution

Assert on a quantity that doesn't vanish under symmetry. Three good
choices:

1. **Variance of projection** — a non-trivial projection has non-zero
   variance even when the centroid is zero:

   ```rust
   let var_c1 = projected.column(0).iter().map(|v| v * v).sum::<f64>() / 8.0;
   assert!(var_c1 > 1e-4, "component should have non-trivial variance");
   ```

2. **Pairwise separation** — verify that points in different rings
   are farther apart in the projection than points in the same ring.

3. **Rank-based test** — verify that a linear classifier can achieve
   non-trivial accuracy on the projected labels.

Used option (1) because it's the smallest assertion change and
directly verifies that the kernel captured *some* structure.

## Prevention

1. **Check assertion sanity against the null hypothesis.** Before
   committing a test, ask: "what would this assertion look like on
   completely random data?" If it would still pass or still fail, the
   assertion isn't measuring what you think.

2. **Recognize symmetry in test fixtures.** Symmetric point clouds
   (centered at origin, rotationally symmetric, etc.) produce
   symmetric projections. Any test that assumes asymmetric output
   will fail.

3. **Prefer variance/distance-based assertions** over centroid-based
   assertions for dimensionality-reduction tests. Variance is a
   second moment and survives symmetric averaging.

## Related

- crates/ix-unsupervised/src/kernel_pca.rs — fixed test
- commit be29905 — original Kernel PCA commit with the fix
