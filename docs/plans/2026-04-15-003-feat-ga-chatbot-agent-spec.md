# ga-chatbot Agent Spec

**Status:** Draft v2 (rewrite — v1 was castle-on-sand)
**Date:** 2026-04-16
**Owner:** spareilleux

## What went wrong with v1

V1 designed an elaborate QA pipeline for a chatbot that couldn't answer basic questions. The chatbot searched static JSON by string matching. That's fundamentally broken. GA already has a 216-dimension OPTIC-K embedding schema and real music theory computation in C#. The chatbot should use those, not reinvent chord theory in Rust.

## Architecture

```
User: "Drop-2 voicings for Cmaj7 on guitar"
    │
    ▼
LLM (GPT-4o / Claude) with tool use
    │ calls GA tools as needed:
    │
    ├── GaParseChord("Cmaj7")
    │   → {root: C, quality: maj7, intervals: [0,4,7,11]}
    │
    ├── GA OPTIC-K embedding search
    │   query: {quality: maj7, symbolic: drop-2, instrument: guitar}
    │   → 216-dim vector → cosine similarity against voicing index
    │   → top-k real voicings with fret diagrams + scores
    │
    ├── GaChordIntervals("Cmaj7")
    │   → interval analysis, voice leading options
    │
    ├── verify_voicing(frets, instrument)
    │   → physical playability check
    │
    └── GaDiatonicChords(key, mode)
        → for progression context (ii-V-I etc.)
    │
    ▼
Grounded answer with REAL voicings from GA's engine
```

Three brains, clear responsibilities:

1. **LLM** — conversational layer. Translates natural language to tool calls. Formats results. NEVER invents voicings.
2. **GA MCP server** — music theory computation. Parses chords, computes intervals, generates OPTIC-K embeddings, searches voicing index. Knows WHAT a Cmaj7 is.
3. **IX MCP server** — structural analysis + validation. Clusters voicings into families, maps fretboard topology, computes shortest voice-leading paths, parses progression grammar, attributes significance via Shapley. Knows HOW voicings relate to each other. Also: adversarial QA and governance gating.

## OPTIC-K Integration

GA's existing embedding schema (v1.6, 216 dimensions):

| Partition | Dims | Weight | What it captures |
|-----------|------|--------|-----------------|
| IDENTITY | 0-5 | filter | Object type (voicing/scale/etc) |
| STRUCTURE | 6-29 | 0.45 | Pitch-class set, ICV, consonance |
| MORPHOLOGY | 30-53 | 0.25 | Fretboard geometry, fingering, span |
| CONTEXT | 54-65 | 0.20 | Harmonic function, voice leading tendency |
| SYMBOLIC | 66-77 | 0.10 | Tags: Drop-2, shell, Hendrix, etc. |
| EXTENSIONS | 78-95 | info | Register, spread, density |
| SPECTRAL | 96-108 | info | DFT phase geometry |
| MODAL | 109-148 | 0.10 | Modal flavors |

Reference: `ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs`

When user asks "Drop-2 Cmaj7 on guitar":
1. GA builds a 216-dim query vector: STRUCTURE from Cmaj7 pitch classes, SYMBOLIC with drop-2 flag, MORPHOLOGY filtered to guitar
2. Cosine similarity against pre-indexed voicing embeddings
3. Returns top-k matches with real fret positions, NOT hallucinated ones

## GA MCP Tools (already exist)

From `ga/GaMcpServer/Tools/`:

| Tool file | What it does |
|-----------|-------------|
| `GuitaristProblemTools.cs` | Core chord/voicing queries |
| `InstrumentTool.cs` | Instrument specs (strings, tuning, range) |
| `KeyTools.cs` | Key signatures, diatonic functions |
| `ScaleTool.cs` | Scale degrees, modes |
| `ModeTool.cs` | Modal interchange |
| `ChordAtonalTool.cs` | Set class analysis |
| `AtonalTool.cs` | Pitch class operations |
| `ContextualChordsTool.cs` | Context-aware chord suggestions |
| `ChatTool.cs` | Existing chat wrapper |
| `SceneControlTool.cs` | Prime Radiant scene control |
| `GaDslTool.cs` | GA domain-specific language |
| `GaScriptTool.cs` | Script evaluation |

The chatbot calls these via MCP (stdio JSON-RPC), same as Claude Code does today.

## Implementation

### Phase 1: Wire GA + IX MCP to the chatbot (3 days)

The ga-chatbot Rust HTTP server spawns BOTH MCP servers as child processes and routes tool calls to the right one.

```
ga-chatbot HTTP server (Rust, port 7184)
    │
    ├── OpenAI/Claude API (tool use loop, up to 5 rounds)
    │
    ├── GA MCP child process (C#, stdio JSON-RPC)
    │   ├── GaParseChord, GaChordIntervals
    │   ├── GaDiatonicChords, GaChordSubstitutions
    │   ├── GaEasierVoicings, GaSearchTabs
    │   ├── GetAvailableInstruments, GetTuning
    │   └── ... (40+ GA tools)
    │
    └── IX MCP child process (Rust, stdio JSON-RPC)
        ├── ix_kmeans (voicing clustering)
        ├── ix_topo (fretboard topology)
        ├── ix_search (A* voice leading)
        ├── ix_graph (transition costs)
        ├── ix_grammar_search (progression parsing)
        ├── ix_stats (corpus profiling)
        ├── ix_governance_check (answer validation)
        └── ... (61 ix tools)
```

What to build:
1. `ga-chatbot serve --http 7184` spawns both child processes at startup
2. Sends `tools/list` to each, merges tool catalogs (prefix: `ga__` / `ix__`)
3. Converts merged schemas to OpenAI function-calling format
4. `execute_tool` routes by prefix to the right child process
5. LLM tool-use loop runs up to 5 rounds, can mix GA + IX calls in one turn

Example flow for "smoothest transition from Dm7 to G7 on guitar":
1. LLM calls `ga__GaParseChord("Dm7")` → intervals, pitch classes
2. LLM calls `ga__GaParseChord("G7")` → intervals, pitch classes
3. LLM calls `ga__GaEasierVoicings("Dm7", instrument="guitar")` → candidate voicings
4. LLM calls `ga__GaEasierVoicings("G7", instrument="guitar")` → candidate voicings
5. LLM calls `ix__ix_search(from=Dm7_voicing, to=G7_voicing, cost="finger_movement")` → A* path
6. LLM formats: "Move from x-5-7-5-6-5 to 3-2-0-0-0-1, total movement cost: 4.2"

### Phase 2: OPTIC-K memory-mapped voicing index (4 days)

#### Dimension: 228 (v1.7, NOT 216)

OPTIC-K schema v1.7 produces 228-dim vectors. The v1.6 (216-dim) references in benchmarks are stale.

#### File format: `state/voicings/optick.index`

```
Header (40+ bytes):
  magic:               [u8; 4] = "OPTK"
  version:             u32 = 2
  endian_marker:       u16 = 0xFFFE
  dimension:           u32 = 228
  count:               u32 = N
  instruments:         u8 = 3
  instrument_offsets:  3 × (offset: u32, count: u32)
  partition_weights:   228 × f32 (sqrt-scaled)

Vectors (N × 228 × f32):
  contiguous, pre-L2-normalized, pre-sqrt-weight-scaled
  sorted by instrument (guitar → bass → ukulele)

Metadata (N × variable):
  msgpack: {diagram, instrument, midiNotes, quality_inferred}
```

100K voicings × 228 × 4 bytes = **87MB**. Memory-mapped read-only via `memmap2`. Brute-force cosine <5ms. No ANN until 5M+ voicings.

#### Partition weighting

Vectors pre-scaled by `sqrt(weight)` per partition before L2-normalization at build time:

| Partition | Dims | Weight | sqrt(w) |
|-----------|------|--------|---------|
| IDENTITY | 0-5 | filter | — |
| STRUCTURE | 6-29 | 0.45 | 0.671 |
| MORPHOLOGY | 30-53 | 0.25 | 0.500 |
| CONTEXT | 54-65 | 0.20 | 0.447 |
| SYMBOLIC | 66-77 | 0.10 | 0.316 |
| EXTENSIONS | 78-95 | info | — |
| SPECTRAL | 96-108 | info | — |
| MODAL | 109-148 | 0.10 | 0.316 |
| HIERARCHY | 149-156 | info | — |
| ATONAL_MODAL | 157-227 | — | — |

Both corpus vectors and query vectors MUST use identical scaling.

#### GA gaps (must be built before IX can consume)

| Gap | What GA needs | Est |
|-----|--------------|-----|
| `Voicing → ChordVoicingRagDocument` bridge | `ChordVoicingRagDocumentFactory.FromVoicing(voicing, analysis)` populating all ~50 fields | 0.5d |
| MCP embedding tool | `ga_generate_voicing_embedding(diagram, instrument)` → 228-dim float[] | 0.5d |
| Bulk export CLI | `FretboardVoicingsCLI --export-embeddings` → binary `optick.index` | 1d |
| Drop-2 structural inference | Verify `MusicalEmbeddingGenerator` infers drop-2 from voicing intervals, not manual tags | 0.5d |
| CLI analysis fields | `--export` must run `VoicingDecomposer` + analysis, not just raw frets | 0.5d |

#### IX deliverable: `crates/ix-optick`

New crate with:
- `OptickIndex::open(path)` — mmap the binary file, validate header
- `OptickIndex::search(query: &[f32; 228], instrument: Option<Instrument>, top_k: usize) -> Vec<SearchResult>`
- `embed(pitch_classes: &[u8], intervals: &[u8], tags: &[&str]) -> [f32; 228]` — query-time embedding in Rust (eliminates GA runtime dep)
- MCP tool: `ix_optick_search`
- Deps: `memmap2`, `serde`, `rmp-serde` (msgpack)

#### Index rebuild safety (Windows)

Windows `CreateFileMapping` holds the file lock. Safe update:
1. GA writes to `optick.index.tmp`
2. Chatbot: `RwLock<Option<Mmap>>` wrapping the index handle
3. On rebuild signal: write-lock → unmap old → rename `.tmp` → `.index` → re-map → unlock

#### Instrument pre-filtering

File sorted by instrument. Header stores 3 `(offset, count)` pairs. Search scans only the target instrument's slice — saves 66% compute.

#### Build order

1. **GA**: `--export-embeddings` CLI + `Voicing → embedding` pipeline (gating prerequisite)
2. **IX**: `crates/ix-optick` mmap loader + cosine search + MCP tool
3. **GA**: MCP tool `ga_generate_voicing_embedding` for real-time query embedding
4. **IX**: `ix_optick::embed()` in Rust for query-time GA independence

Phase 2a (steps 1-2) gets the chatbot working with real embeddings.
Phase 2b (steps 3-4) removes GA runtime dependency for queries.

### Phase 3: Adversarial QA with Octopus (2 days)

NOW the QA pipeline makes sense — it validates a chatbot that actually works:
1. Send graduated prompts to the chatbot
2. Chatbot calls GA tools, returns grounded answers
3. Octopus dispatches answers to 3 judge personas
4. Judges verify: did the LLM correctly translate the GA tool results? Did it hallucinate anything beyond what GA returned?
5. Hexavalent aggregation, Shapley attribution

The QA tests the LLM's translation accuracy, not its music theory knowledge (GA handles that).

## What GA provides (music theory computation)

- Chord parsing, interval computation — `GaParseChord`, `GaChordIntervals`
- OPTIC-K 216-dim embeddings for semantic voicing search
- Instrument specs (string count, tuning, range) — `GetAvailableInstruments`, `GetTuning`
- Diatonic analysis, modal interchange — `GaDiatonicChords`, `GetModeChords`
- Voicing generation and fretboard layout — `GaEasierVoicings`, `GaSearchTabs`
- Chord substitutions — `GaChordSubstitutions`, `GaSetClassSubs`

## What IX provides (structural analysis + validation)

| IX tool | Chatbot capability it enables |
|---------|------------------------------|
| `ix_kmeans` | "Show me voicings similar to this one" — cluster-based nearest-neighbor |
| `ix_topo` | "Are these voicings in the same neighborhood?" — persistent homology on fretboard space |
| `ix_search` (A*) | "Smoothest transition from Dm7 to G7" — minimal finger-movement path |
| `ix_graph` | "Which chord changes are hardest?" — transition cost graph |
| `ix_grammar_search` | "Is this a valid jazz progression?" — CFG parse over chord sequences |
| `ix_game_nash` | "What's the optimal voicing choice here?" — game-theoretic analysis |
| `ix_stats` | "How common is this voicing pattern?" — statistical profiling of the corpus |
| `ix_governance_check` | "Is this answer grounded?" — constitutional compliance gate |
| `ix_sanitize` | Input sanitization before LLM |
| `ix_adversarial_fgsm` | Adversarial robustness testing of the chatbot |

GA knows WHAT a chord is. IX knows HOW chords relate to each other structurally.

## What the LLM provides (conversation, not computation)

- Natural language understanding
- Tool call orchestration (decides which GA + IX tools to call)
- Result formatting and explanation
- Multi-turn context

## MVP scope

**Phase 1 only.** Wire GA MCP to the chatbot. No OPTIC-K yet (Phase 2), no QA pipeline yet (Phase 3). Just: user asks → LLM calls GA tools → real answer.

Success criteria: "Drop-2 voicings for Cmaj7 on guitar" returns voicings computed by GA's C# engine, not hallucinated by the LLM.

## Kill criteria

- If GA MCP server startup takes >10s, the child-process approach is too slow. Fall back to pre-built GA executable.
- If GA doesn't expose a voicing-search tool, add one to GaMcpServer before proceeding.
- If LLM ignores tool results and hallucinates anyway, add a post-processing step that strips any voicing not returned by a tool call.

## Effort

- Phase 1: 2 days (Rust HTTP ↔ C# MCP plumbing)
- Phase 2: 3 days (OPTIC-K embedding pipeline + Qdrant index)
- Phase 3: 2 days (Octopus QA wiring, already partially built)
- Total: 7 days

## What we already built (reusable)

- `crates/ga-chatbot` — HTTP server, CLI, QA harness, aggregation module (keep all of this)
- `crates/ix-sanitize` — input sanitization (keep)
- `crates/ix-governance` hexavalent extension (keep)
- Adversarial corpus — 77 prompts across 8 categories (keep, expand)
- `.claude/skills/adversarial-qa/SKILL.md` — Octopus skill (keep)
- `.github/workflows/adversarial-qa.yml` — CI workflow (keep)
- React frontend at `/chatbot` (keep, just fix the model chip)

## What we throw away

- `search_voicings` Rust function (static JSON grep — fundamentally wrong)
- `parse_chord_pitch_classes` Rust function (reinventing what GA already does)
- The idea that ix provides answers (it provides validation)
