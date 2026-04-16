# Session Plan — OPTIC-K Three-Brain Chatbot

**Created:** 2026-04-16
**Intent Contract:** See .claude/session-intent.md

## What You'll End Up With

A working three-brain chatbot where GA computes real 228-dim OPTIC-K embeddings, IX searches them via memory-mapped index, and the LLM orchestrates tool calls — no hallucinated voicings, no static JSON grep, no toys.

## How We'll Get There

### Phase Weights
- Discover: 10% — Verify GA DI wiring, confirm VoicingAnalyzer produces all fields needed by VoicingDocumentFactory
- Define: 15% — Spec-lock binary format contract, write format validation test, confirm partition weights match
- Develop: 45% — Build across GA (CLI + binary writer + MCP tool) and IX (ix-optick crate + chatbot wiring)
- Deliver: 30% — Cross-repo integration test, benchmark 3 instruments, Windows mmap rebuild smoke test

### 🐙 DEBATE CHECKPOINTS IN THIS PLAN:

🔸 After Define phase: "Is the binary format contract solid? Any endianness, alignment, or versioning issues?"
   Triggers: 1-round adversarial debate (Claude + Codex + Gemini) on format risks

🔸 After Develop phase: "Does the cross-repo pipeline produce correct embeddings end-to-end?"
   Triggers: 1-round collaborative debate on edge cases (empty voicings, single-note, atonal)

### Execution Strategy — 5 Parallel Agents

| Agent | Repo | Deliverable | Deps |
|-------|------|------------|------|
| 🔴 **GA-CLI** (Codex) | ga | `--export-embeddings` mode in FretboardVoicingsCLI: DI setup, VoicingAnalyzer pipeline, binary index writer | None |
| 🟡 **GA-MCP** (Gemini) | ga | `VoicingEmbeddingTool.cs` in GaMcpServer: real-time embedding generation via MCP | None |
| 🔵 **IX-Optick** (Claude) | ix | `crates/ix-optick`: mmap loader, header validation, brute-force cosine search, MCP tool | GA-CLI (needs format spec) |
| 🟢 **IX-Chatbot** (Claude) | ix | `crates/ga-chatbot/src/mcp_bridge.rs`: child process spawning, tool catalog merge, LLM tool-use loop | GA-MCP + IX-Optick |
| 🟤 **Format-Contract** (OpenCode) | both | Binary format spec doc + validation tests in both C# and Rust | None (runs first) |

### Build Order

```
Phase 1 (parallel):
  🟤 Format-Contract: spec-lock optick.index binary format
  🔴 GA-CLI: add GA.Business.ML ref, DI, --export-embeddings mode
  🟡 GA-MCP: VoicingEmbeddingTool.cs

Phase 2 (after format contract):
  🔵 IX-Optick: mmap loader + cosine search + MCP tool (needs format spec)

Phase 3 (after GA-MCP + IX-Optick):
  🟢 IX-Chatbot: mcp_bridge.rs wiring both servers

Phase 4 (integration):
  All: cross-repo smoke test with real voicings
```

### Execution Commands
To execute this plan, run:
```
/octo:embrace "Build OPTIC-K three-brain chatbot"
```

Or execute phases individually:
- `/octo:discover` — verify GA DI, confirm VoicingAnalyzer fields
- `/octo:define` — spec-lock binary format, write validation tests
- `/octo:develop` — build all 5 deliverables
- `/octo:deliver` — cross-repo integration, benchmarks, mmap safety

## Provider Requirements
🔴 Codex CLI: Available ✓ — GA-CLI agent (C# heavy lifting)
🟡 Gemini CLI: Available ✓ — GA-MCP agent (MCP tool creation)
🟤 OpenCode: Available ✓ — Format contract agent (cross-repo spec)
🟢 Ollama: Available ✓ — Local fallback
🔵 Claude: Available ✓ — IX-Optick + IX-Chatbot agents (Rust work)

## Success Criteria
1. `FretboardVoicingsCLI --export-embeddings --tuning guitar` produces valid optick.index
2. `ga_generate_voicing_embedding("x-3-2-0-1-0", "guitar")` returns 228 floats via MCP
3. `ix-optick` mmaps the file, searches "Cmaj7 drop-2 guitar" and returns real voicings
4. Chatbot at port 7184 answers "Drop-2 voicings for Cmaj7 on guitar" with GA-computed results
5. All 3 instruments work (guitar 24 frets, bass 21, ukulele 15)
6. Windows mmap rebuild (write .tmp → unmap → rename → re-map) doesn't crash

## Next Steps
1. Review this plan
2. Adjust if needed (re-run /octo:plan)
3. Execute with /octo:embrace when ready
