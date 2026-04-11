# Lens pattern + Ledger/Tableau/Purview vocabulary — forward-looking notes

**Status:** late-arriving insights, captured for next session
**Source:** Claude paradox-hunter subagent from the 2026-04-10 multi-AI brainstorm on AgentContext/AgentAction. The subagent was dispatched in parallel with Codex + Gemini but completed ~hours after the implementation shipped; the synthesis at the time used only Codex + Gemini because two providers is the Team-mode gate. This doc captures what the subagent added when it finally returned.
**Scope:** NOT a redesign. Primitives 1–4 are already shipped and working against the current `ix-agent-core` substrate. This doc is evolution notes for when primitives 5–7 land and start feeling cramped in the current design.

---

## What the subagent validated (already shipped)

Five core framings from the subagent match what's running on `origin/main` today:

1. **Transforms ARE Events.** `MiddlewareVerdict::Replace` emits `SessionEvent::ActionReplaced`. The dispatcher re-projects from the log. Middleware never mutates `ReadContext` directly.
2. **Middleware chains don't fold verdicts algebraically.** `MiddlewareChain::dispatch` runs middlewares in sequence with `Continue`/`Block`/`Replace` early returns. No `Hexavalent::or`-style composition — consistent with the OR decision doc.
3. **Pre-dispatch type ≠ handler view.** `WriteContext<'a>` wraps a `&'a ReadContext` plus a `&'a mut dyn EventSink`; middleware sees both, handlers see only the read half. Matches the Anthropic read-only vs LangChain mutable resolution.
4. **Bit-exact HashMap sorting is structural.** Every `BTreeMap<String, _>` in `ReadContext` is deliberate — the governance-instrument contract requires cross-process replay identity.
5. **Explicit composition over algebraic.** Verdict composition is by sequence and early return, not by algebraic laws over a truth-value type.

None of these need new work. They're shipped and tested.

---

## Three ideas worth keeping for next session

### 1. The `(Event, Projection, Verdict)` triple as the primitive meta-pattern

The subagent observed: every shipped harness primitive *is* a triple of (an event kind, a state projection, a verdict type).

- **`ix-context`**: event = `ContextWalked`, projection = `ContextBundle`, verdict = implicit (bundle completeness)
- **`ix-loop-detect`**: event = recorded tool call, projection = sliding window state, verdict = `LoopVerdict::TooManyEdits`
- **`ix-approval`**: event = classified action, projection = `ApprovalVerdict.rationale`, verdict = `Tier` + `MiddlewareVerdict::Block`
- **`ix-session`**: event = `SessionEvent`, projection = on-disk log, verdict = n/a (pure persistence)

The template isn't currently named as the unifying shape. Making it explicit would:

- Give primitives 5–7 (fuzzy, trace flywheel, advanced MCP) a ready-to-fill template rather than re-inventing the shape each time
- Let us write a `trait HarnessPrimitive` that captures the contract, with associated types for `Event`, `Projection`, `Verdict`. Instead of five ad-hoc middleware impls, every primitive gets a default middleware adapter from the trait.
- Reveal when a proposed primitive doesn't fit the shape (which would be a design smell — maybe it belongs in the substrate or it's not really a primitive).

**Not an immediate refactor.** Try applying the template to `ix-fuzzy` (primitive #5) when that ships and see if the abstraction earns its keep. If all three of {fuzzy, flywheel, advanced MCP} fit cleanly, formalize the trait.

### 2. Ledger / Tableau / Purview vocabulary

Current code uses "session log" / "ReadContext" / "metadata slice." The subagent proposed:

- **Ledger** — the full append-only event history. Today: `SessionLog` in `ix-session`.
- **Tableau** — the projection visible at a specific turn (current ordinal). Today: `ReadContext` in `ix-agent-core`.
- **Purview** — the narrow slice a specific handler is permitted to see. Today: the `metadata: BTreeMap<String, JsonValue>` inside `ReadContext`, consulted per-tool.

The words are better than what we have. "Ledger" makes the append-only commitment audible; "tableau" captures the per-turn projection without dragging in "context" which is overloaded; "purview" names the per-tool slice without needing a full type.

**Not an immediate rename.** Rust rename-refactors across five crates and ~215 tests are expensive. Adopt the vocabulary in new docs and new primitives first — if it sticks, do a single big rename commit when momentum naturally reopens those files. If the words feel forced after 2–3 primitives, drop them.

Worth adopting now: use Ledger/Tableau/Purview in the next brainstorm docs to normalize the vocabulary before it's load-bearing.

### 3. Lens trait instead of monolithic ReadContext

The subagent's strongest structural idea: rather than one `ReadContext` struct with everything middleware/approval/session/handlers might want, define a `Lens` trait:

```rust
pub trait Lens {
    fn project(ledger: &Ledger, ordinal: u64) -> Self;
}

pub struct MiddlewareLens { /* fields relevant to middleware chain */ }
pub struct ApprovalLens   { /* fields relevant to approval classification */ }
pub struct ToolLens       { /* fields relevant to a specific tool handler */ }
pub struct SessionLens    { /* fields relevant to session-level reasoning */ }

impl Lens for MiddlewareLens { fn project(ledger: &Ledger, ordinal: u64) -> Self { ... } }
// ... one impl per primitive
```

Each lens is a deterministic function of the ledger up to a specific ordinal. The lens types are **pure data**, derived on demand. No shared monolithic struct; no god-object drift as primitives pile on.

**Evaluation vs current design:**

- The current `ReadContext` is already a lens conceptually — it's `f(EventLog, ordinal) → ReadContext` per the design doc thesis. What's missing is the trait abstraction that lets multiple lens types coexist.
- For 4 primitives, one `ReadContext` works fine. For 7+ primitives, each wanting slightly different projection shapes, it starts to bloat.
- The refactor is mechanical: pull `ReadContext` fields into the relevant per-primitive lenses, add the `Lens` trait, have middleware dispatch choose a lens per downstream consumer.
- **Blocker:** the current code has `AgentHandler::run(&ReadContext, &AgentAction)`. Refactoring to `AgentHandler::run(&impl Lens, &AgentAction)` is a breaking API change across every consumer.

**Not an immediate refactor.** But when primitive #5 (ix-fuzzy) lands and wants fields that don't belong in the generic `ReadContext`, consider this. If `ReadContext` starts accumulating primitive-specific fields (like fuzzy confidence projections, trace flywheel aggregates), the `Lens` trait is the escape hatch.

### Smaller ideas — parked for now

- **"Proposal" vs "Commitment"** terminology — declared intent vs effective reality. Current code uses "action" for both. Worth adopting in naming once something breaks in a way that surfaces the distinction. For MVP, not worth the churn.
- **Content-addressed capability handles via BLAKE3** — replacing runtime UUIDs with BLAKE3 hashes of declaration events. Elegant for replay (handles re-derive from the ledger automatically), but adds a crypto dependency and solves a problem we don't have yet (nothing in primitives 1–4 uses UUIDs for capabilities).
- **"Invocation" instead of "AgentAction"** — naming nitpick. "Action" is fine.

---

## What NOT to do with this doc

- Do not treat it as a design doc that blocks further implementation. Primitives 5–7 can ship against the current substrate without any of these changes.
- Do not retroactively refactor primitives 1–4 to adopt the Lens trait or Ledger/Tableau/Purview vocabulary. The shipped design works. Refactor when pressure from new primitives actually forces it.
- Do not treat the subagent's late return as a signal the shipped design is wrong. The subagent explicitly validates 5/5 core framings. The three novel ideas are evolution paths, not corrections.

## When to consult this doc

- When `ix-fuzzy` (primitive #5) starts and you're wondering what shape its middleware should take — try the `(Event, Projection, Verdict)` template.
- When `ReadContext` accumulates a field that only one consumer cares about — consider pulling it into a `FooLens` struct instead.
- When writing documentation for a new primitive — try the Ledger/Tableau/Purview vocabulary and see if it reads better than "log/context/slice."

## Source material

- `docs/brainstorms/2026-04-10-agent-context-action.md` — the synthesis that shipped (Codex + Gemini contributions, Claude subagent deferred)
- This session's final implementation: `ix-agent-core`, `ix-approval`, `ix-session` on `origin/main` at commit `bfaf6fb` or later
- The late subagent output itself, which proposed the triple pattern, the three-word vocabulary, and the Lens trait
