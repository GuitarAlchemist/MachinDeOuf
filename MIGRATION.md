# Migration: MachinDeOuf → ix

**Date:** 2026-03-14
**Tag before rename:** `v0.1.0-machin-final`

## What changed

The project has been renamed from **MachinDeOuf** to **ix** (pronounced "icks").

All crate names changed: `machin-*` → `ix-*`
All Rust module paths changed: `machin_*` → `ix_*`
All MCP tool names changed: `machin_*` → `ix_*`
CLI binary: `machin` → `ix`
MCP server binary: `machin-mcp` → `ix-mcp`

## Why

"ix" is shorter, cleaner, and reflects the project's identity as a machine-learning forge.
The rename was planned as part of the broader repo-split strategy (ix / tars / ga / Demerzel).

## If you had the old repo

GitHub redirects `GuitarAlchemist/MachinDeOuf` → `GuitarAlchemist/ix` automatically.
Update your local remote:

```bash
git remote set-url origin https://github.com/GuitarAlchemist/ix.git
```

## Rollback

```bash
git checkout v0.1.0-machin-final
```
