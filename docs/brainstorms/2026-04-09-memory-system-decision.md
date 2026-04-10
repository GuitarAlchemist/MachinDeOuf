---
date: 2026-04-09
topic: memory-system-decision
status: decision
---

# Decision: no claude-mem for ix; enhance existing memory systems instead

## Context

Considered adopting [claude-mem](https://github.com/thedotmack/claude-mem)
as a persistent memory layer for ix sessions. It provides automatic
capture via lifecycle hooks, SQLite+Chroma storage, and semantic search.

## Decision

**Do not install claude-mem.** It duplicates existing systems and
violates ix's "git-tracked, reviewable, pure dependencies" philosophy.

## Rationale

ix already has three complementary memory layers:

1. **`~/.claude/projects/.../memory/MEMORY.md`** — per-user auto-memory,
   loaded every session. Captures user preferences, project state,
   feedback patterns.
2. **`docs/solutions/`** — git-tracked institutional knowledge via
   `/ce:compound`. Shared across team, searchable via grep.
3. **`docs/brainstorms/` + `docs/plans/`** — session artifacts
   committed to the repo.

claude-mem would add a fourth system with:
- Binary SQLite + Chroma storage (not git-trackable)
- Local-only persistence (no team sharing)
- Heavy runtime deps (Bun, Python, uv, Chroma)
- No cross-repo benefit to tars or ga

The philosophical mismatch is the dealbreaker: ix values deterministic,
reviewable outputs, and an opaque binary memory store fails that test.

## Ideas worth stealing from claude-mem

File these as follow-up enhancements to the existing memory systems:

### 1. SessionEnd auto-compound prompt
Add a hook (or CLAUDE.md instruction) that at session end reviews
whether non-trivial work was done and suggests running `/ce:compound`
before closing. Catches insights that would otherwise be lost.

### 2. Progressive disclosure for docs/solutions lookups
Enhance `learnings-researcher` agent to return frontmatter-only
results first (title + tags + one-line summary), then fetch full
content only for the top 1-2 matches. Saves ~10x context on common
lookups.

### 3. Semantic frontmatter enrichment
Enrich `docs/solutions/*.md` frontmatter with:
- `related:` list of other solution docs
- `invalidates:` for superseded approaches
- `co_occurs_with:` for patterns that travel together

### 4. SessionStart relevance injection
At session start, scan `docs/solutions/` for docs whose tags overlap
with the initial prompt, and surface the top 3 titles with links in
the CLAUDE.md-loaded context. Makes institutional knowledge
actively consulted, not passively waiting.

### 5. Vector index for docs/solutions (OPTIONAL)
If grep-based search becomes insufficient as the solution library
grows, consider an ix-native vector index built on `ix-math::bsp`
(20-feature similarity) or `ix-gpu::cosine_similarity`. This would
be IX-spirited: pure Rust, deterministic, no external runtime deps.

## Non-goals

- Do not try to replicate claude-mem's automatic capture of every
  tool invocation. That level of detail is noise in ix's workflow;
  the compound-at-milestones pattern is sufficient.
- Do not add SQLite or any binary-storage dependency for memory.
- Do not compete with claude-mem on its merits — our use case is
  different (git-first ecosystem) and they serve their users well.
