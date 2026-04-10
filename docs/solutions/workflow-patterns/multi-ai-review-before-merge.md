---
title: "Multi-AI review should happen BEFORE merge, not after"
category: workflow-patterns
date: 2026-04-09
tags: [code-review, multi-ai, octopus, workflow, merge-before-review]
symptom: "Session dispatched 6 parallel agents -> merged their work -> then ran multi-AI review and found 10 issues, all already committed"
root_cause: "Review is cheap to run on isolated worktrees but expensive to apply as follow-up commits after merge; flow should have reviewed-then-merged, not merged-then-reviewed"
---

# Review-before-merge for parallel agent flows

## Problem

Ran a Team-of-Teams parallel flow:

```
1. Brainstorm -> Plan
2. 6 parallel worker agents in isolated worktrees
3. Merged all 6 worktrees into main
4. /octo:review on the merged commits
5. Found 10 issues (2 HIGH, 8 MEDIUM)
6. 9 follow-up fix commits
```

The review found real bugs: LDA non-symmetric deflation (silent data
corruption), Sedenion log(-1) (broken round-trip), 4x duplicated
eigensolver, NaN propagation, OOM vectors. All of these were already
committed to main before the review ran.

The follow-up fix phase was **9 more atomic commits** — roughly the
same amount of work as the original feature implementation. The fixes
would have been much cheaper if applied inside each worker's worktree
before merge.

## Root cause

Reviewing committed code is expensive because:

1. **History pollution**: Every fix becomes a new commit. Easy to end
   up with 2x the commit count for 1x the feature.
2. **Diff noise**: Reviewers can't cleanly see which bug lives in
   which worker's original work.
3. **Integration cost**: Fixes may conflict with each other because
   they share files (e.g., multiple fixes to the same module).
4. **Lost isolation**: Once merged, the worker's sandbox is gone — you
   can't easily re-dispatch the same worker to fix its own work with
   full context of what it originally did.

The cheap path is to review each worker's output **while the worker's
worktree still exists**, then either:
- Feed fixes back to the same worker for a second pass
- Gate merge on the reviewer's sign-off
- Or discard the work and respawn the worker with a better spec

## Working solution (for future flows)

Proposed orchestration pattern:

```
1. Brainstorm -> Plan -> Spec (formal interface contracts)
2. Prep commit (shared types, scaffolding)
3. For each work package:
   a. Spawn worker agent in worktree
   b. Worker implements + commits to its branch
   c. Spawn REVIEWER agent against the worker's branch
      (diff = worktree-branch..main)
   d. Reviewer returns findings
   e. If HIGH findings: feed back to worker, re-review
   f. If MEDIUM findings: queue as follow-up work but don't block merge
   g. If no blockers: mark branch as ready-to-merge
4. Sequential merge phase (fast-forward each ready branch)
5. Post-merge: full workspace test, one integration commit if needed
6. Compound learnings
```

This session's commits `789cf54`, `2c2b8f7`, `9ebbe01`, `ec62595`,
`c34176e`, `4df0cda`, `67abe77`, `435b462`, `e6057b4` would all have
been inlined into the original worker commits instead of appearing
as follow-up fixes.

## Prevention

1. **Default the orchestrator to review-before-merge.** Require an
   explicit flag to skip review for fast iteration.

2. **Review agents should see the WORKER'S branch**, not a merged diff.
   This gives them the full context of what the worker was trying to
   accomplish without other workers' changes as noise.

3. **Budget review explicitly.** If review is expected to take 20% of
   implementation time, plan for it up front. Don't let it be
   "skipped because we're running long."

4. **Track fix-after-merge commits as a workflow metric.** High counts
   mean the review-before-merge gate was skipped or is ineffective.

## Related

- docs/solutions/workflow-patterns/parallel-worktree-merge-pitfalls.md
  — related lessons from the same session
- /octo:review skill — the reviewer that caught these bugs
- Commits 789cf54..e6057b4 — the 9 follow-up fixes that would have
  been avoidable with review-before-merge
