# Session Intent Contract

**Created:** 2026-04-16
**Session:** OPTIC-K Three-Brain Chatbot Build

## Job Statement

Build the complete three-brain chatbot pipeline: GA produces 228-dim OPTIC-K embeddings via CLI + MCP, IX consumes them via mmap index + cosine search, chatbot wires GA + IX MCP servers as child processes for grounded music theory conversation.

## Success Criteria

1. **Working binary index** — `FretboardVoicingsCLI --export-embeddings` produces `optick.index` with real 228-dim vectors for guitar, bass, ukulele
2. **MCP embedding tool** — `ga_generate_voicing_embedding(diagram, instrument)` returns 228-dim float[] via GA MCP server
3. **IX can consume it** — `ix-optick` crate mmaps the binary, validates header, runs brute-force cosine search <5ms
4. **Production-ready** — All 3 instruments, Windows-safe mmap rebuild, partition weighting matches between GA build-time and IX query-time

## Boundaries

- Use existing GA infrastructure: `EmbeddingSchema.cs`, `MusicalEmbeddingGenerator`, `VoicingDocumentFactory`, `VoicingAnalyzer`
- Binary format is the contract between GA and IX — must be spec-locked before coding
- No ANN indexing until 5M+ voicings — brute-force cosine is sufficient for 100K
- IX crate is standalone once the binary file exists — no GA runtime dependency for search

## Context

- **Knowledge level:** Expert — deep design review completed, all GA source files read
- **Scope clarity:** Fully specified — Plan 003 has header format, partition weights, GA gaps, build order
- **Constraints:** High stakes (wrong format = garbage index), cross-repo coordination (GA produces, IX consumes)
- **Reference:** `docs/plans/2026-04-15-003-feat-ga-chatbot-agent-spec.md`

## Build Order (from Plan 003)

1. GA: `--export-embeddings` CLI + `Voicing → ChordVoicingRagDocument` bridge + binary writer
2. IX: `crates/ix-optick` mmap loader + cosine search + MCP tool
3. GA: MCP tool `ga_generate_voicing_embedding` for real-time query embedding
4. IX: `ix_optick::embed()` in Rust for query-time GA independence
5. GA+IX: Chatbot HTTP server spawns both MCP servers, merges tool catalogs, LLM tool-use loop

## GA Gaps Identified

| Gap | Deliverable | Est |
|-----|------------|-----|
| CLI skips analysis in export mode | `--export-embeddings` runs VoicingAnalyzer + VoicingDocumentFactory + MusicalEmbeddingGenerator | 1d |
| No binary index writer | `OptickIndexWriter.cs` — header + vectors + msgpack metadata | 0.5d |
| No MCP embedding tool | `VoicingEmbeddingTool.cs` in GaMcpServer | 0.5d |
| DI not wired in CLI | Add GA.Business.ML reference, minimal ServiceProvider | 0.5d |
| Drop-2 structural inference | Verify MusicalEmbeddingGenerator infers from intervals, not manual tags | 0.5d |
