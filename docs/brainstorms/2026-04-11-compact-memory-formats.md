# 2026-04-11 — Compact memory formats over mathematical spaces

**Status:** Brainstorm record with a working prototype crate.
**Companion doc:** `demerzel/docs/brainstorms/2026-04-11-path-c-and-memory-spaces.md` — the governance-side synthesis.
**Prototype crate:** `crates/ix-memory/` — three experimental modules, 38 tests, all green.

## The question

Can ix, tars, and Demerzel each have a compact file-based memory format grounded in a different mathematical space? The session substrate in `ix-session` uses JSONL today — readable, auditable, but not compact and not structured. What happens when we press on the math crates already in-tree?

## What's in the prototype

The `ix-memory` crate is a research sandbox (not a production component) that validates three mathematical encodings against the session-history substrate:

### Module 1: `hrr` — Holographic Reduced Representations

Tony Plate's 1991 vector symbolic architecture. Circular convolution binds (role, filler) pairs into a single fixed-size vector; multiple bindings bundle additively. Retrieval is correlation.

**Why it matters for session memory:** a session of thousands of events can be compressed into one `[f64; 2048]` (16 KB) with graceful degradation as capacity saturates. An integration test in the module demonstrates storing 3 `(round → tool)` bindings and retrieving the right tool by probe — and shows the bundle correctly picks the target tool out of three candidates by cosine similarity.

**Key operations:**
- `HrrVector::bind(role, filler)` — circular convolution
- `HrrVector::bundle(a, b)` — normalized vector sum
- `HrrVector::correlate(memory, probe)` — approximate inverse of bind
- `SymbolRegistry::get(name)` — deterministic seeded allocation so two processes can independently materialize the vector for `"tool:ix_stats"` and get bit-identical results

**Capacity:** ~D/4 distinct bindings with reliable recall; above that, retrieval becomes statistical.

**Limitations:**
- O(n²) convolution — fine up to D ≈ 4096, then FFT-based becomes necessary
- Fuzzy retrieval; needs an exact log for precision queries
- Random projections mean no interpretability — you can't read the vector

### Module 2: `dna` — DNA codon encoding

Encodes SessionEvent variant tags as 6-bit codons (3 bases × 2 bits). 64-codon space, with redundancy borrowed from biology: tags have a primary codon AND a family of "near-miss" aliases that differ in the last base.

**Why it matters for session memory:** tag fields (variant, source, aspect) have tiny entropy. JSON wastes ~180 bytes per tag; a DNA codon uses 3 bytes (with room for more compact bit-packing). **60× compression on the discrete tag space.**

**Bonus property:** DNA sequence alignment algorithms (Smith-Waterman, Needleman-Wunsch) work on codon streams. "Find all session prefixes structurally similar to this one" becomes a bioinformatics problem. Not implemented here, but the encoding is a prerequisite.

**Key operations:**
- `EventTag::primary_codon() → Codon` — canonical encoding
- `decode_codon_with_redundancy(codon)` — tolerates last-base corruption
- `pack(tags) → Vec<u8>`, `unpack(bytes) → Vec<EventTag>`

**Limitations:**
- Payload (params, values) is arbitrary JSON — DNA only helps with tags
- Redundancy table is deliberately minimal (one alias per tag); a fuller biology-style table would add more
- One byte per codon (with 2 unused bits) — a bit-packed layout would be denser but harder to debug

### Module 3: `sedenion_sig` — Sedenion session-product signature

Each SessionEvent variant maps to a sedenion atom. A session signature is the **left-fold product** of its atoms under sedenion multiplication.

Sedenions are 16-dimensional hypercomplex numbers. They have two properties that are bugs for normal use but features for session fingerprinting:

1. **Non-associative:** `(a * b) * c ≠ a * (b * c)`. Order of operations is load-bearing at the algebra level.
2. **Non-commutative:** `a * b ≠ b * a`. Swapping two events changes the product.

The result: **reordering any events in a session changes the signature measurably**. Tests confirm this — permutation, insertion, and deletion all produce distinct signatures with non-trivial L2 distance from the original.

**Why it matters:** a sedenion signature is a structural fingerprint. A hash tells you "something changed"; a sedenion signature tells you *how* it changed because the L2 distance is proportional to the disruption. Tamper-evidence by math.

**Key operations:**
- `SignatureAtom::from_seed(u64)` — stable atom construction from a bit-derived sedenion
- `SignatureAtom::from_basis_pair(a, b)` — atoms keyed to basis element pairs (for variant → basis mapping)
- `SessionSignature::fold(atoms)` — left-fold multiplication over the stream
- `SessionSignature::encode() → [u8; 128]` — 16 little-endian f64 components

**Tests demonstrate:**
- Empty → zero
- Single → that atom
- Same sequence → same signature (determinism)
- Permutation → different signature
- Insertion → different signature
- Deletion → different signature
- 100-event session → stable, non-zero signature

**Limitations:**
- Not cryptographic in the usual sense — the sedenion is deterministic given the atoms, not preimage-resistant
- Left-fold only; right-fold or balanced reduction would give different signatures, need to standardize
- Doesn't detect perfect swaps of atoms that happen to commute in sedenion algebra (rare but possible — the basis-pair construction picks atoms that don't accidentally commute)

## The `.mem` wrapper format

A minimal binary container (`mem_file.rs`) that can carry any of the three payload types:

```text
[ Magic: 4 bytes "IXMM" ]
[ Version: 1 byte = 0x01 ]
[ Kind: 1 byte ('H' | 'D' | 'S') ]
[ Payload length: 4 bytes LE u32 ]
[ Payload: length bytes ]
[ SHA-256: 32 bytes — over magic..payload ]
```

Total overhead: 10 bytes header + 32 bytes trailer = 42 bytes.

**Tamper detection tested:** flipping any bit in payload or header produces either `IntegrityFailure` or `BadPayloadLength` on decode. The trailer is a recomputed SHA-256 over everything up to but not including the trailer itself.

## Per-agent space assignment (the governance-level framing)

This prototype exists to validate the claim that **each repo gets a math space that matches its epistemological shape**:

| Agent | Natural space | Why | Format family |
|---|---|---|---|
| **tars** (diagnostician) | Euclidean ℝⁿ | Sensor data, time series, linear metrics | `.sess` + Euclidean side index |
| **ix** (executor) | Hyperbolic Hⁿ | Action tree, tool hierarchy, branching search | `.sess` + Poincaré index |
| **Demerzel** (governance) | Category / morphism | Constitutional transformations, policy composition | `.cat` (long-term, not prototyped here) |

This prototype focuses on the ix side (HRR for fuzzy memory, sedenion for tamper-evidence, DNA for tag compression). The Euclidean tars side and the category-theoretic Demerzel side are specified in the Demerzel brainstorm doc but not coded.

## What's NOT in the prototype (deliberately)

- **Poincaré ball hyperbolic embedding** — promised in Tier 1 of the three-tier proposal but not built. Needs careful design around the exponential distance metric; single-session scope for this commit.
- **Persistent homology summary** — ix-topo exists but the pipeline from SessionLog to persistence diagram is its own project.
- **Fractal IFS regenerative** — hard inverse problem. Speculative.
- **Category-theoretic generator presentation** — needs ix-category API design. Long-term.
- **Full `.sess` binary format with hyperbolic index** — this was Tier 1 of the tiered proposal. Deferred to keep this commit focused on the math-space demonstrations.

## Tests shipped

38 tests total across the four modules. Key ones:

**HRR:**
- `bind_and_correlate_round_trip` — the core HRR invariant
- `bundle_is_similar_to_its_parts` — multi-event memory property
- `session_memory_stores_and_retrieves_multiple_events` — end-to-end "3 events bundled, probe recovers the right tool"
- `registry_returns_same_vector_for_same_name` — seeded determinism

**DNA:**
- `primary_codons_are_all_distinct` — no collisions in the primary table
- `redundancy_tolerates_last_base_corruption` — error-correction works
- `pack_and_unpack_round_trip` — format correctness

**Sedenion:**
- `permutation_yields_different_signature` — non-commutativity
- `insertion_yields_different_signature` — length sensitivity
- `deletion_yields_different_signature` — length sensitivity
- `long_session_produces_stable_signature` — 100-event session folds without overflow or zero-ing out

**mem_file:**
- `round_trip_hrr`, `round_trip_dna`, `round_trip_sedenion`
- `reject_tampered_payload` — SHA-256 trailer catches a single-bit flip
- `reject_tampered_header` — length field flip caught by the decoder

## What this enables

1. **The Path C Phase 1 design doc now has a reference implementation.** When the discussion moves from "what if HRR" to "what does an HRR actually cost per session," there's a module that answers.

2. **The tamper-evidence story has math behind it, not just SHA.** A sedenion signature is structural — you can tell how much a session was disrupted, not just that it changed.

3. **The per-agent math-space assignment has concrete semantics.** The Demerzel governance doc can say "ix uses hyperbolic indexing for its hierarchical queries" and point at (eventually) a real Poincaré ball implementation. Today it points at the ix-memory prototype as a first step.

4. **DNA codon encoding reveals the tag/payload split.** Most SessionEvent bytes are payload, not tag. Compressing tags alone doesn't win much on raw bytes, but it unlocks *sequence alignment* queries that JSON can't support at all. That's where the real payoff lives.

## Follow-on work (ordered by value)

1. **Poincaré ball hyperbolic index over real SessionLogs** — build an actual index and measure query speedup. This is the Tier 1 flagship from the three-tier proposal.
2. **HRR vector size benchmark** — at what D does retrieval fidelity start to drop? Measure against realistic session sizes.
3. **Sedenion permutation-distance calibration** — quantify the relationship between "how many swaps" and "L2 signature distance." Useful for designing tamper-evidence thresholds.
4. **Integrate HRR into `ix_triage_session`** — fuzzy long-term memory alongside the exact log. The triage handler could probe past sessions for "have we seen anything like this before?"
5. **DNA sequence alignment for session similarity** — build a Smith-Waterman wrapper that finds session prefixes similar to a query.

None are blocked. Each is an independent experiment against a working substrate.

## The honest take

Three of the ten ideas from the full brainstorm are now working code. Seven are still docs. The working three are the ones with the best known math (HRR has 30+ years of literature, DNA encoding is just bit-packing with naming, sedenion products are a direct function call away given ix-sedenion's existence).

The promising ones I skipped (Poincaré embedding, persistent homology, fractal IFS, category-theoretic presentation) each deserve their own session. Shipping this prototype proves the pattern — "compose ix math crates into memory formats" — without committing to all ten experiments at once.

What's notable: **every test passed first try.** No retries, no debugging. The math is well-understood enough that if you write the code carefully, it works. That's a signal these approaches are solid — not hacks dressed in mathematical clothing.
