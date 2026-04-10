---
title: "TARS v1 is the right quarry for IX-spirited exotic math"
category: ecosystem-integration
date: 2026-04-09
tags: [tars, ecosystem, porting, sedenion, hyperbolic, exotic-math]
symptom: "IX needed inspiration for next-round features beyond the obvious ML staples"
root_cause: "TARS v2 is explicitly deferring exotic math (sedenions, hyperbolic, Hurwitz quaternions, CUDA) to v3+; that deferred territory is exactly ix's mission"
---

# TARS v1 as an IX-spirited math quarry

## Context

ix and tars are sibling projects in the GuitarAlchemist ecosystem (ix
is Rust, tars is F#). Both share governance (Demerzel) and a math
orientation. In early 2026, tars is in the middle of a v1 -> v2
re-architecture that explicitly **defers** a lot of exotic math to v3+
because it's "too complex for the pragmatic v2 scope."

Quote from `v2/docs/4_Research/V1_Insights/v1_component_reusability_analysis.md`:

> "Defer to v3+: Hyperbolic embeddings, Sedenions & exotic math DSLs"
>
> "Advanced Mathematics -- DEFER -- Explicitly v3+ per v2 docs"

**That deferred territory is exactly ix's mission.** ix already has
ix-sedenion, ix-math::poincare_hierarchy, ix-math::bsp, ix-rotation,
ix-gpu — the pieces tars v2 chose not to port. So tars v1 becomes a
useful quarry: pick the components tars v2 is throwing away and see
which ones fit ix's philosophy.

## High-value picks (this session)

### 1. Unified GeometricSpace enum (TARS HyperComplexGeometricDSL.fs)

TARS v1 had a single `GeometricSpace` discriminated union wrapping 10
distance metrics behind one dispatch function:

```fsharp
type GeometricSpace =
    | Euclidean
    | Hyperbolic of curvature: float32
    | Spherical of radius: float32
    | Minkowski of signature: int * int * int * int
    | Mahalanobis
    | Wasserstein
    | Manhattan
    | Chebyshev
    | Hamming
    | Jaccard
```

ix had all the building blocks scattered across `ix-math::distance`,
`ix-math::hyperbolic`, and ad-hoc reimplementations in downstream
crates. Porting the unified enum + `distance()` dispatcher took ~350
LOC and gives every downstream consumer (clustering, KNN, embeddings)
a single API to swap geometries. Shipped as `ix-math::geometric_space`
in commit `085490f`.

### 2. Sedenion exp/log via scalar+vector decomposition

TARS v1 implements sedenion exponential and logarithm using the
quaternion trick generalized to 16 dimensions:

```text
exp(a + v) = exp(a) * (cos|v| + (v/|v|) * sin|v|)
log(r * (cos theta + u sin theta)) = log(r) + u * theta
```

ix-sedenion had basic arithmetic but no transcendental functions.
Porting was ~50 LOC. Enables sedenion-based Lie group operations,
exponential maps, hypercomplex gradient descent. Shipped as new
methods on `Sedenion` in commit `2a06af4`.

(Note: TARS v1's version has the same bug I later had to fix around
`log(-1)` — see docs/solutions/math-correctness/sedenion-log-negative-reals.md)

### 3. 16D sedenion BSP partitioner (TARSSedenionPartitioner.fs)

TARS v1 builds BSP trees where split planes are sedenion-valued
normals, not axis-aligned. Each hyperplane carries a "Significance"
score enabling importance-weighted partitioning. ix-math::bsp uses
axis-aligned splits; porting the hyperplane-oriented variant is in
the planned next round (not yet shipped).

## Lower-value or deferred

- **Hurwitz Quaternions**: ix-rotation already has quaternions; integer
  lattice quaternions are useful for number theory but niche.
- **CUDA kernels**: ix-gpu uses WGPU which is cleaner cross-platform.
- **Metascript executor**: ix-pipeline covers this.
- **FLUX multi-language DSL**: too F#-specific.
- **Fractal grammars**: ix-grammar already has CFG/Earley/CYK.

## Pattern: how to mine TARS v1 for ix

1. **Look at what v2 is explicitly deferring.** Their "defer to v3+"
   list is your priority queue.
2. **Filter for pure-math, zero-external-deps components.** Those are
   the easiest to port and match ix's philosophy.
3. **Check if ix already has the pieces but not the unified API.**
   Often the port is about dispatch patterns, not new algorithms.
4. **Verify TARS v1 tests before porting.** Ports often inherit bugs
   (see the sedenion log case).
5. **Add a doc comment crediting the TARS v1 source file.** Gives
   future maintainers a reference for the algorithm origin.

## Prevention / future work

- Keep watching tars v2 as it evolves. Every "defer to v3+" decision
  is a potential ix port opportunity.
- Consider a `federation-mine-tars` skill that periodically grep-mines
  tars v1 for components matching ix's domain list.
- When porting from TARS, add property tests that verify the
  quaternion / complex / real limits behave correctly. TARS v1 doesn't
  always test these and can ship subtly-wrong math.

## Related

- crates/ix-math/src/geometric_space.rs — first port
- crates/ix-sedenion/src/sedenion.rs — second port (exp/log)
- docs/brainstorms/2026-04-09-tars-v1-inspirations.md — full analysis
- docs/solutions/math-correctness/sedenion-log-negative-reals.md —
  bug inherited from TARS v1 and fixed in ix
