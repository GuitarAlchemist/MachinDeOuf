---
title: "Parallel worktree merge pitfalls when agents write to main"
category: workflow-patterns
date: 2026-04-09
tags: [git, worktree, parallel-agents, orchestration, team-of-teams]
symptom: "Parallel worker agents return as 'completed' but shared files conflict and some agents write to main instead of their isolated worktree"
root_cause: "Worker agents use cargo and shared target/ directories, don't commit on return, and can accidentally resolve paths to the main working tree"
---

# Parallel worktree merge pitfalls

## Problem

Launched 6 parallel `Agent` calls with `isolation: "worktree"` to implement
independent phases of the Code Observatory. All 6 reported success but:

1. **Phase 5's worker wrote to `main` instead of its worktree** — the agent
   reported all its test results successfully, but the worktree was empty
   and the files showed up on main as untracked.
2. **All 5 remaining worktree branches had `0` commits beyond the base** —
   the changes existed only in the worktree's working directory, not
   committed to the branch. `git worktree list` showed each at the base
   commit.
3. **Shared `target/` directory hit 37 GB / 100% disk** because each worker
   ran `cargo build` and `cargo test` without isolating its target dir.
4. **Shared files `Cargo.toml` and `lib.rs` had 6 conflicting diffs** that
   had to be hand-merged.
5. **4 different workers duplicated the same helper** (`jacobi_symmetric_eigendecomp`)
   because they had no way to share code.

## Root cause

Git worktrees share the underlying object database but each worker's
agent session is a black box. Without explicit protocol:

- Workers don't `git commit` before returning — they leave dirty working
  directories, and the orchestrator has no transaction semantics.
- `cargo` uses `CARGO_TARGET_DIR` unless explicitly overridden, and the
  workspace-level `target/` is shared across all worktrees by default.
- Path resolution via `CARGO_MANIFEST_DIR` or `std::env::current_dir()`
  can silently escape the worktree, especially if an agent tool does
  any path canonicalization.
- Shared files need hand-merging because each worker edited them in
  isolation.

## Working solution

Harvest pattern for this session:

```bash
# 1. For each worktree, cp the worker's new module files to main
for wt in ae158589 a32dec27 a71e1b14 a394bde8 ac891866; do
  cp .claude/worktrees/agent-$wt/crates/ix-code/src/*.rs \
     crates/ix-code/src/
done

# 2. Manually merge 6 Cargo.toml feature stanzas into one
# 3. Manually merge 6 lib.rs module declarations into one
# 4. Run full workspace test to verify integration
cargo test -p ix-code --features full
```

## Prevention

For future parallel-work flows, enforce these as orchestrator protocol:

1. **Interface-first prep commit**: Before dispatching workers, land one
   commit containing shared types, trait definitions, and scaffolded
   module stubs with empty `pub mod` lines. Workers only fill in bodies.
   Zero merge conflicts.

2. **Atomic commit-or-abort**: Each worker must `git add -A && git commit`
   before returning. If it doesn't commit, the orchestrator treats the
   work as failed and does NOT attempt to harvest dirty working-dir state.

3. **Disk budget enforcement**: Before dispatching, check `df` and cap
   parallel workers to keep usage < 80%. Workers should use
   `CARGO_TARGET_DIR=/tmp/worker-N/target` to avoid collision.

4. **Declared file ownership**: Each worker declares its `scope:` list
   (files it may modify). Attempts to touch files outside scope are
   logged. Shared files (Cargo.toml, lib.rs) go to a merge queue.

5. **Cross-worker learnings broadcast**: When multiple workers
   independently solve the same subproblem (e.g. Jacobi eigensolver),
   a coordinator agent reviewing drafts before finalization should flag
   duplicates and route them to a common helper.

## Related

- docs/solutions/feature-implementations/ — other session fix docs from
  the same day
- docs/solutions/workflow-patterns/multi-ai-review-before-merge.md
