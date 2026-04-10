---
title: "Windows LNK1318 PDB size limit when linking test binaries with many features"
category: build-errors
date: 2026-04-09
tags: [windows, msvc, linker, pdb, cargo-features, rust]
symptom: "LINK : fatal error LNK1318: Unexpected PDB error; LIMIT (12) 'ix_code-XXXXXXXX.pdb'"
root_cause: "MSVC linker PDB size limit (~4 GB for the standard format) exceeded when building a test binary with all feature-gated modules enabled simultaneously"
---

# Windows LNK1318 PDB size limit

## Problem

Running `cargo test -p ix-code --features full` where `full` enables
all 6 feature-gated modules (semantic, trajectory, topology, gates,
advanced, physics) failed with:

```
LINK : fatal error LNK1318: Unexpected PDB error; LIMIT (12)
  'C:\Users\...\target\debug\deps\ix_code-9f854c03bfe59758.pdb'
```

Each feature pulls in its own transitive dependency tree (tree-sitter,
git2/libgit2, ix-topo, ix-governance, ix-types, ix-chaos, ix-signal,
ix-graph, ix-math, ix-ktheory, ix-fractal). The combined symbol count
in the unit-test binary exceeded the MSVC debug info format limit.

## Root cause

Windows MSVC's default PDB format (`PDB 7.0`) has a hard limit on the
total number of symbols and debug info records it can store. On debug
builds, every generic monomorphization, every inlined function, and
every trait vtable adds symbols. When you enable many features
simultaneously in a workspace with ~30 crates, the test binary can
easily exceed this limit even though each feature in isolation works
fine.

This is NOT:
- A Rust bug
- A Cargo bug
- Specific to ix-code
- Reproducible on Linux (uses DWARF, no such limit)

It IS:
- A fundamental MSVC debug format limit
- Triggered by the sum of all enabled feature code, not any single
  feature
- Known to the Rust compiler team (issue rust-lang/rust#98302 and
  related)

## Working solution

Three workarounds, in order of preference:

**1. Test features in groups instead of all at once**

```bash
# Safe: each group stays well under the PDB limit
cargo test -p ix-code --features "semantic,trajectory,topology"
cargo test -p ix-code --features "gates,advanced,physics"
cargo test -p ix-code  # default (layer 1 only)
```

This is what ix uses in practice. Each group tests a logically related
subset of layers.

**2. Use release mode (strips debuginfo)**

```bash
cargo test -p ix-code --features full --release
```

Release builds have `debug = false` by default, so the PDB is tiny.
Downside: loses test-time debug symbols, slower build, slower
optimization-level assertions may behave differently.

**3. Use `debug = 1` (line info only, no types)**

Add to root `Cargo.toml`:

```toml
[profile.test]
debug = 1
```

Cuts PDB size dramatically by dropping type info but keeping line
numbers. Stack traces still work for test failures.

## Prevention

1. **Don't assume `cargo test --features full` will always work on
   Windows.** Document the feature-group test pattern in the crate
   README.

2. **Check PDB size if you add a large new dependency.** If it's near
   the limit, consider whether the new feature should be in its own
   feature group or opt-in test target.

3. **CI should test feature groups separately on Windows,** not as one
   monolithic `--all-features` command. Linux CI can still use
   `--all-features` because DWARF has no such limit.

4. **Avoid heavy generic code in hot paths** when possible. Each
   monomorphization adds PDB entries. Concrete types or dynamic
   dispatch cut symbol count.

## Related

- Rust issue rust-lang/rust#98302 — MSVC debug info format limits
- Microsoft docs: https://learn.microsoft.com/en-us/cpp/build/reference/pdbpagesize
  — `/PDBPAGESIZE` can raise the limit but requires Visual Studio 17.6+
  and may break older tools
