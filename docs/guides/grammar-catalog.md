# The ix grammar catalog

> What it is, when to use it, how to chain it into `ix_grammar::ebnf::parse` or `ix_grammar::abnf::parse`.

`ix_grammar_catalog` is a curated, MCP-queryable index of real-world grammar sources across EBNF, ABNF, PEG, ANTLR G4, W3C EBNF, and BNF notations. It answers agent questions like "where is the authoritative Python grammar" or "which RFC defines the URI ABNF" with a canonical URL, the notation format, and enough metadata to pick the right downstream parser. It does **not** host the grammar files themselves — see the authoritative source for every entry.

## What's in it

Thirty-ish entries, grouped into four clusters:

**Programming languages.** Python PEP 617, Go language specification, ECMAScript (TC39), Rust reference, Ruby (whitequark/parser), Haskell 2010 report, R7RS Scheme, C11 (ISO/IEC 9899:2011), and the WebAssembly core specification.

**Data formats.** JSON (RFC 8259), TOML v1.0.0, YAML 1.2, CSS3 Syntax Module, SQL-2016 (from the community-maintained EBNF extraction), and GraphQL June 2018.

**IETF protocol grammars (all ABNF).** HTTP/1.1 (RFC 9112), HTTP Semantics (9110), URI (3986), TLS 1.3 (8446), DNS (1035), SMTP (5321), IMAP4rev2 (9051), MIME (2045), OAuth 2.0 (6749), and WebSockets (6455).

**Meta-grammars and aggregators.** ABNF (RFC 5234, self-referential), ISO/IEC 14977 EBNF (self-referential), the ANTLR `grammars-v4` repository (~200 G4 grammars), GitHub Linguist's vendor grammars, and the two canonical Rust parser libraries `pest` (PEG) and `nom` (parser combinators).

## How to query it

```json
// From an MCP client, via ix_grammar_catalog
{ "language": "rust" }
{ "format": "abnf" }
{ "topic": "rfc" }
{ "language": "http", "format": "abnf" }
```

All three filter fields (`language`, `format`, `topic`) are optional and AND-combined. Omitting all of them returns every entry. Language matching is case-insensitive; entries with `language: "many"` (the meta-pointers) always pass.

## Chaining into the parsers

The catalog's whole point is that every entry's `format` field tells you which ix-grammar parser (if any) can consume it natively:

- **`format: "ebnf"`** — use [`ix_grammar::ebnf::parse`](../../crates/ix-grammar/src/ebnf.rs). Supports the ISO 14977 subset that real specs actually use (rule definition with `=` or `::=`, `|` alternation, `[...]` optional, `{...}` repetition, `(...)` grouping, quoted literals, `(* ... *)` comments).

- **`format: "abnf"`** — use [`ix_grammar::abnf::parse`](../../crates/ix-grammar/src/abnf.rs). Supports the RFC 5234 subset: `=` and `=/` rule definition, `/` alternation, `[...]` optional, `(...)` grouping, `*element` / `1*element` / `N*M element` repetition, quoted literals, `; line comments`.

- **`format: "w3c_ebnf"`** — not yet natively supported. Use the ISO EBNF parser for the simpler bits and fall back to a dedicated library for the regex-style postfix operators (`+`, `*`, `?`).

- **`format: "peg"`** — not supported natively. Use [`pest`](https://pest.rs/) directly; the catalog's pest entry is the pointer.

- **`format: "antlr_g4"`** — not supported natively. Use ANTLR itself, or one of the Rust ANTLR bridges. The `grammars-v4` entry is the authoritative source of G4 files.

- **`format: "bnf"`** — no native parser. Most BNF grammars are small enough to rewrite in EBNF by hand; the catalog's BNF entries are historical references.

## End-to-end example

From the inside of a Rust crate consuming ix-grammar:

```rust
use ix_grammar::abnf;
use ix_grammar::catalog::{self, GrammarFormat};

// Find the HTTP Semantics entry.
let http_entries = catalog::by_language("http");
let http_rfc = http_entries
    .iter()
    .find(|e| e.format == GrammarFormat::Abnf && e.name.contains("9110"))
    .expect("HTTP Semantics entry");
println!("Fetch the grammar from {}", http_rfc.url);

// Your code fetches the ABNF file and stores it in `body: String`.
// Once you have it:
let body = "# ... contents of RFC 9110 ABNF extracted from the text ...";
let grammar = abnf::parse(body)?;
assert!(!grammar.productions.is_empty());
```

`grammar` is an `ix_grammar::constrained::EbnfGrammar`, which plugs directly into the existing grammar-guided MCTS state, the weighted-rule machinery in `ix_grammar::weighted`, and the replicator dynamics in `ix_grammar::replicator`.

## Supported subset vs full standards

Both parsers target the subset of their notations that appears in real-world specs, not the full ISO / IETF standards. Features you won't find:

- EBNF: exception rules (`-`), special sequences (`? ... ?`), explicit repetition counts.
- ABNF: numeric value ranges (`%x41-5A`), strict length-exact repetition bounds.

For full coverage, use dedicated third-party parsers. The ix-grammar subset was chosen to match what the catalog's own entries need to be parseable.

## Adding a new entry

Append a `GrammarEntry { ... }` literal to `CATALOG` in [`crates/ix-grammar/src/catalog.rs`](../../crates/ix-grammar/src/catalog.rs), preserving the rough grouping by cluster. The `every_entry_is_well_formed` unit test will reject anything with an empty field, a URL that doesn't start with `http://` or `https://`, or a year before 1960. If the entry's language or format deserves a new query path, add a test case to `python_query_returns_python_grammar` or `abnf_format_query_catches_the_rfc_protocol_block` patterns.

## References

- [`crates/ix-grammar/src/catalog.rs`](../../crates/ix-grammar/src/catalog.rs) — the catalog source of truth
- [`crates/ix-grammar/src/ebnf.rs`](../../crates/ix-grammar/src/ebnf.rs) — ISO 14977 EBNF parser (subset)
- [`crates/ix-grammar/src/abnf.rs`](../../crates/ix-grammar/src/abnf.rs) — RFC 5234 ABNF parser (subset)
- [`crates/ix-grammar/src/constrained.rs`](../../crates/ix-grammar/src/constrained.rs) — the target `EbnfGrammar` type
- [`docs/guides/code-analysis-tools.md`](code-analysis-tools.md) — the sibling code-analysis catalog guide
- [`docs/guides/rfc-catalog.md`](rfc-catalog.md) — the sibling RFC catalog guide
