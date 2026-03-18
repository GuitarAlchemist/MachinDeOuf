---
title: "Karnaugh maps for tetravalent logic minimization"
category: feature-implementation
date: 2026-03-18
tags: [karnaugh, tetravalent-logic, governance, policy-minimization, demerzel]
components:
  - crates/ix-governance/karnaugh.rs
  - crates/ix-governance/tetravalent.rs
related_prs: [19]
---

# Karnaugh Maps for Tetravalent Logic

## Problem

Demerzel governance policies involve multiple conditions (action risky? evidence contradictory? agent authorized?) each with 4 possible truth values {T, F, U, C}. Complex multi-condition policies needed a way to be simplified to minimal rule sets — classical Karnaugh maps extended to 4-valued logic.

## Solution

### New: `ix-governance/karnaugh.rs`
- `KarnaughMap` — 1-4 variable maps over {T, F, U, C}, 4^n cells
- `prime_implicants()` — find maximal groups via recursive constraint relaxation
- `minimal_cover()` — greedy set cover for smallest rule set
- `Implicant` with named variable display
- `display()` / `truth_table()` — formatted output

### Extended: `TruthValue`
- `implies()` — A → B ≡ ¬A ∨ B
- `xor()` — exclusive or
- `equiv()` — biconditional A ↔ B
- `is_definite()` / `is_indefinite()` — classify T/F vs U/C
- `all()` — canonical iteration

### CI fixes (pre-existing)
- `ix-dashboard/state.rs` — added `#[allow(dead_code)]` for serde-only struct fields
- `ix-dashboard/reader.rs` — `map_or(false, ...)` → `is_some_and(...)` (nightly clippy)
- `ix-governance/violation_pattern.rs` — `sort_by` → `sort_by_key` (nightly clippy)

## Use Cases
- Governance policy simplification ("when does Demerzel say Escalate?")
- PSAP dispatch decision minimization
- Agent capability routing optimization

## Cross-References
- [PR #19](https://github.com/GuitarAlchemist/ix/pull/19)
- [ix-governance/tetravalent.rs](../../crates/ix-governance/src/tetravalent.rs) — foundation
