# The ix RFC catalog

> What it is, what's in it, how to walk the obsolescence graph.

`ix_rfc_catalog` is a curated, MCP-queryable index of the IETF RFCs that define the modern internet stack — IP through HTTP/3, TLS 1.3, DNS + DNSSEC, OAuth, JOSE, SMTP, IMAP, SSH, and the BCPs that ground the rest. It exists so agents doing protocol work can answer "what's the current spec for X" and "what did RFC X replace" with a single MCP call instead of trawling through `rfc-editor.org`. The catalog is **not** a mirror of the full ~9,000-RFC index — see [rfc-editor.org](https://www.rfc-editor.org/rfc-index.html) for that. ix covers the curated subset that actually shows up in agent work.

## Scope (explicit)

**In scope:** ~70 RFCs across IP, transport, HTTP, TLS / crypto / security, DNS, email, realtime media, data formats, SSH, NTP, plus the BCPs that every spec cites (BCP 14 MUST/SHOULD, BCP 47 language tags), meta-grammar RFCs (ABNF RFC 5234), and a handful of historical entries including RFC 1.

**Out of scope:** RFCs that have never been widely cited, experimental-track drafts, IETF organizational memos, April 1 RFCs, historical RFCs without a direct current-stack descendant, and protocol extensions that haven't achieved standard status.

## Obsolescence graph

Every entry has `obsoletes: &[u32]` and `obsoleted_by: &[u32]` fields that wire the entry into the graph of who-replaced-whom. The relationship is symmetric within the catalog: if RFC A lists B in its `obsoletes`, then B lists A in its `obsoleted_by`. This invariant is enforced by a unit test that runs on every `cargo test`, so the graph stays consistent as the catalog grows.

The graph's primary value is **catching stale citations**. Agents routinely quote RFC 2616 when the current HTTP specification is RFC 9110 + 9111 + 9112 (the 2022 consolidation), because 2616 is what they were trained on. The `current_standard` filter and the `obsolescence_chain` query together give agents a clean path from "I know about 2616" to "the live spec is 9110."

## Querying

```json
// From an MCP client, via ix_rfc_catalog
{ "number": 9110 }                              // exact lookup
{ "topic": "http" }                             // all HTTP-tagged
{ "topic": "http", "current_standard": true }   // drops obsoleted
{ "status": "internet_standard" }               // filter by status
{ "obsolescence_chain": 2616 }                  // walk both directions
```

Filter fields are optional and AND-combined, except for `obsolescence_chain` which overrides the others and returns the complete chain for the seed RFC.

## Two worked examples

**Example 1 — "What's the current HTTP spec?"**

```json
{ "topic": "http", "current_standard": true }
```

Returns a short list that includes 9110 (semantics), 9111 (caching), 9112 (HTTP/1.1), 9113 (HTTP/2), 9114 (HTTP/3), 6455 (WebSockets), 3986 (URI). Crucially, it does **not** include 2616, 7230, 7231, 7232, 7233, 7234, 7235, or 7540 — all of which are in the catalog but flagged as `Obsoleted`. The `current_standard` filter walks each entry's `obsoleted_by` field and drops anything non-empty.

**Example 2 — "What did RFC 2616 become?"**

```json
{ "obsolescence_chain": 2616 }
```

Returns every RFC in the connected component of 2616's obsolescence graph: 2616 itself, the 2014 split (7230, 7231, 7232, 7233, 7234, 7235), and the 2022 consolidation (9110, 9111, 9112). The agent can now quote the current spec with confidence and cite the replacement chain for historical context.

## What status really means

The catalog uses a simplified RFC status vocabulary:

- **`internet_standard`** — on the standards track at the highest level. RFC 791 (IPv4), RFC 1034 + 1035 (DNS), RFC 768 (UDP), RFC 9110 (HTTP Semantics), RFC 9293 (TCP).
- **`proposed_standard`** — widely implemented and stable, not yet promoted to Internet Standard. Most modern RFCs land here: TLS 1.3 (8446), OAuth 2.0 (6749), DoH (8484).
- **`draft_standard`** — a maturity level that was retired by RFC 6410 in 2011. Some older RFCs still carry it (SMTP RFC 5321 is a notable example).
- **`experimental`** — experimental track; not a standard.
- **`informational`** — documents the state of the art or provides guidance without defining a standard. DMARC RFC 7489 and the WebRTC overview RFC 8825 live here.
- **`obsoleted`** — superseded by a later RFC. The replacement is in the `obsoleted_by` array. Use `current_standard: true` to exclude these from queries.

## Adding a new entry

Append an `RfcEntry { ... }` literal to `CATALOG` in [`crates/ix-net/src/rfc_catalog.rs`](../../crates/ix-net/src/rfc_catalog.rs), preserving the rough grouping by topic. Three hard requirements:

1. **Valid URL.** Must start with `https://www.rfc-editor.org/rfc/rfc` and contain the RFC's own number. The `every_entry_is_well_formed` test enforces this.

2. **Obsolescence symmetry.** If the new RFC obsoletes an existing catalog entry, update that entry's `obsoleted_by` field. If the new RFC is itself obsoleted by an existing entry, update that entry's `obsoletes` field. The `obsolescence_graph_is_symmetric_for_recorded_entries` test will fail hard on any inconsistency. Cross-check against the actual RFC text on rfc-editor.org — the official "Obsoletes" header is the source of truth.

3. **No duplicates.** The test suite rejects duplicate RFC numbers. If you need to model jointly-obsoleted relationships (e.g. RFC 7230 is jointly obsoleted by 9110 and 9112), list both obsoleters in the single 7230 entry's `obsoleted_by` array.

## References

- [`crates/ix-net/src/rfc_catalog.rs`](../../crates/ix-net/src/rfc_catalog.rs) — the catalog source of truth
- [`crates/ix-catalog-core/src/lib.rs`](../../crates/ix-catalog-core/src/lib.rs) — shared `Catalog` trait
- [rfc-editor.org](https://www.rfc-editor.org/rfc-index.html) — the authoritative full RFC index
- [`docs/guides/grammar-catalog.md`](grammar-catalog.md) — sibling catalog (RFC 5234 ABNF appears in both)
- [`docs/guides/code-analysis-tools.md`](code-analysis-tools.md) — sibling catalog (external code-analysis tools)
- [`docs/MANUAL.md §4`](../MANUAL.md#4-the-61-mcp-tools--by-category) — where this catalog fits in the broader tool surface
