---
title: "Sedenion log(-1) principal value needs pi on an imaginary axis"
category: math-correctness
date: 2026-04-09
tags: [sedenion, hypercomplex, logarithm, principal-value, quaternion-limit]
symptom: "exp(log(-1)) returned 1 instead of -1, breaking round-trip for negative real sedenions"
root_cause: "Scalar+vector decomposition of log assumed non-zero vector part; for pure-real negative inputs it collapsed to ln(|s|) alone with no angle component"
---

# Sedenion log(-1) principal value

## Problem

Implemented sedenion `exp` and `log` via scalar+vector decomposition
ported from TARS v1:

```rust
pub fn log(&self) -> Sedenion {
    let scalar = self.components[0];
    let vec_norm = /* ... */;
    let norm = self.norm();
    let mut out = [0.0; 16];
    out[0] = norm.ln();
    if vec_norm >= 1e-12 {
        let angle = vec_norm.atan2(scalar);
        let factor = angle / vec_norm;
        for i in 1..16 { out[i] = factor * self.components[i]; }
    }
    // else: fall through with only out[0] set
    Sedenion::new(out)
}
```

Regression:

```rust
let neg_one = Sedenion::from_scalar(-1.0);
let l = neg_one.log();
// l.components == [0, 0, 0, ..., 0]  (because ln(|-1|) = 0)
let back = l.exp();
// back == [1, 0, 0, ..., 0]  instead of [-1, 0, ..., 0]
```

The quaternion limit (sedenion restricted to 4 non-zero components)
also failed — this bug is inherited from any scalar+vector decomp that
doesn't special-case the negative real axis.

## Root cause

For a sedenion `s = a + v` with `|v| = 0`:
- `norm = |a|`
- `out[0] = ln(|a|) = 0` when `a = -1`
- `out[1..16] = 0` because the vector-part branch was skipped

But `log(-1)` is NOT zero. In complex arithmetic, the principal value
`Log(-1) = i * pi`. In hypercomplex algebras the principal log of a
negative real needs to pick a direction along an imaginary axis — any
unit imaginary will do, but conventionally `e_1`.

The scalar+vector decomposition formula is derived assuming `|v| > 0`,
so the "else" branch silently produces garbage on the negative real
axis.

## Working solution

Special-case the negative-real-axis scenario: when `vec_norm` is near
zero AND `scalar < 0`, place the `pi` rotation on `e_1`:

```rust
if vec_norm >= 1e-12 {
    // General case: log has a well-defined vector direction
    let angle = vec_norm.atan2(scalar);
    let factor = angle / vec_norm;
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        *slot = factor * self.components[i];
    }
} else if scalar < 0.0 {
    // Negative-real case: principal log needs angle pi on unit imaginary.
    // Choose e_1 by convention so exp(log(-1)) == -1, matching the
    // complex principal value Log(-1) = i*pi.
    out[1] = std::f64::consts::PI;
}
// Positive-real case with vec_norm ~= 0: out[0] = ln(|s|) is correct
```

After fix:

```rust
let l = Sedenion::from_scalar(-1.0).log();
assert_eq!(l.components[0], 0.0);
assert_eq!(l.components[1], std::f64::consts::PI);
let back = l.exp();
assert!((back.components[0] + 1.0).abs() < 1e-10);
```

## Prevention

1. **Property-test round-trips** on edge cases. Any time you implement
   `exp` and `log`, add `log(exp(x)) ≈ x` and `exp(log(x)) ≈ x` tests
   with inputs covering: positive real, negative real, zero, pure
   imaginary, mixed, and small/large magnitudes.

2. **Compare against the quaternion/complex limit.** Sedenions
   generalize quaternions which generalize complex numbers. If a
   sedenion operation restricted to the first 2 or 4 components doesn't
   match complex/quaternion behavior, the implementation has a bug.

3. **Document principal-value conventions** in the doc comment. The
   choice of `e_1` for the negative-real axis is conventional, not
   derived — future readers should know they can change it if their
   application has a different natural axis.

## Related

- crates/ix-sedenion/src/sedenion.rs — fixed implementation
- commit 2c2b8f7 — the fix
- Multi-AI review finding #4 (Codex GPT-5.4)
- TARS v1 reference: `v1/src/TarsEngine.FSharp.Core/DSL/HyperComplexGeometricDSL.fs`
  — original port source, which had the same bug
