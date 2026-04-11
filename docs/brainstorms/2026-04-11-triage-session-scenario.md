# 2026-04-11 — Triage Session: the first end-to-end harness scenario

**Status:** Design doc, not yet implemented.
**Context:** All 7 harness primitives shipped (see `project_context_dag_wip.md`
resume notes). The roadmap is done at the primitive level. The memory's
resume instructions list **"build an actual agent scenario that exercises
the full harness end-to-end"** as one of the top strategic options. This
doc specifies that scenario as a single new MCP tool.

## Why this tool and not something else

Every primitive that just landed is currently exercised by *unit* and
*integration* tests. None of them has been driven end-to-end by a real
LLM making a real decision that flows through the real dispatcher. A
single well-shaped MCP tool can:

1. Validate that bidirectional sampling, the middleware chain, the
   session log, the trace flywheel, and the fuzzy layer all compose
   without impedance mismatches.
2. Produce a replayable JSONL trace that proves the harness works on
   real — not mocked — LLM output.
3. Give future scenarios a concrete template for "sample → structured
   plan → governed dispatch → learn from own trace."
4. Expose which middlewares fire under realistic prompts, feeding back
   into the OR/hexavalent decision doc.

## Tool shape

```text
name: ix_triage_session
description:
  Analyze recent session activity, ask the client LLM which ix tools
  would advance current investigations, rank the proposed actions by
  hexavalent confidence, dispatch the ranked plan through the governed
  middleware chain, and optionally export the resulting trace back
  through the flywheel for self-improvement.

input_schema:
  focus:       string  (optional free-text hint)
  max_actions: int     (default 3, min 1, max 8)
  learn:       bool    (default true — export trace via flywheel)
```

Routed through `ToolRegistry::call_with_ctx` like `ix_explain_algorithm`,
because it needs `ServerContext` for `sample()`.

## Handler flow

```
fn triage_session_with_ctx(args, ctx: &ServerContext) -> Result<Value, String>:

  // ── 1. Inspect recent session state ──
  events = read_installed_session_log(last_n = 20)
      .ok_or("triage requires an installed SessionLog")?;
  summary = summarize(events);  // tools called, blocks, observations

  // ── 2. Sample the client LLM for a structured plan ──
  system = "You are the ix harness triage agent. Given recent session
            events, propose up to {max_actions} ix tool invocations that
            would advance the investigation. Respond with ONLY a JSON
            array:
              [{\"tool_name\": string,
                \"params\": object,
                \"confidence\": one of ['T','P','U','D','F','C'],
                \"reason\": string}]
            Do NOT propose ix_triage_session (would recurse).
            Available tools: {registry.list_names()}";
  user = f"Recent session:\n{summary}\n\nFocus hint: {focus or 'none'}";

  plan_text = ctx.sample(user, system, 800)?;
  plan = parse_plan_strict(plan_text)
      .or_else(|_| parse_plan_lenient(plan_text))?;  // extract JSON array between first [ and last ]

  // ── 3. Rank by hexavalent confidence ──
  dist = HexavalentDistribution::from_plan(&plan);  // primitive #5
  plan.sort_by_hex_tiebreak(&dist);                 // C > U > D > P > T > F

  // ── 4. Hard recursion guard ──
  plan.retain(|p| p.tool_name != "ix_triage_session");

  // ── 5. Dispatch each action through the governed chain ──
  let cx = ReadContext::synthetic_for_legacy();  // or inherit, when available
  let mut dispatched = Vec::new();
  for (i, item) in plan.iter().enumerate() {
      let action = AgentAction::InvokeTool {
          tool_name: item.tool_name.clone(),
          params: item.params.clone(),
          ordinal: 0,  // dispatch_action assigns
          target_hint: item.params.get("target")
                          .and_then(|v| v.as_str())
                          .map(str::to_string),
      };
      let result = dispatch_action(&cx, action);
      dispatched.push(DispatchResult::from(&item, result));
  }

  // ── 6. Optional self-learning via the flywheel ──
  let trace_dir = if args.learn {
      Some(flywheel::export_current_session()?)  // primitive #6
  } else {
      None
  };
  let ingest = trace_dir
      .as_ref()
      .map(|dir| registry_bridge::dispatch(
          "ix_trace_ingest",
          json!({ "trace_dir": dir })
      ))
      .transpose()?;

  // ── 7. Synthesis ──
  Ok(json!({
      "plan": plan,
      "dispatched": dispatched,        // {tool_name, ok, value|blocked_by}
      "events_emitted": sink.len(),
      "trace_dir": trace_dir,
      "trace_ingest": ingest,
      "model": /* from sampling metadata if exposed */,
  }))
```

## Primitive coverage matrix

| # | Primitive | Exercised how |
|---|---|---|
| 1 | Context DAG (`ix-context`) | Indirectly — LLM may propose `ix_context_walk` targets |
| 2 | Loop detector | Duplicate proposals trip the circuit breaker; result surfaces `ActionBlocked` |
| 2b | Substrate (`ix-agent-core`) | `AgentAction` construction, `ReadContext`/`WriteContext`, `SessionEvent` emission |
| 3 | Approval / blast-radius | Each dispatch runs the tier classifier; Tier 1/2 auto-approves, Tier 3+ blocks |
| 4 | Session log | `install_session_log` required; every verdict + outcome appended to JSONL |
| 5 | Fuzzy (`HexavalentDistribution`) | Ranks LLM's confidence-tagged plan before dispatch |
| 6 | Trace flywheel | `export_current_session` → `ix_trace_ingest` round-trip |
| 7 | MCP sampling | `ctx.sample()` is the whole triage decision |

One MCP call touches every primitive. That is the entire point.

## Open design choices

### 1. Parse failure policy

If the LLM emits unparseable garbage:

- **(a)** Retry once with stricter prompt ("Previous response was not
  valid JSON. Emit ONLY the array.")
- **(b)** Fail the tool with the raw text in the error message
- **(c)** Emit an `EmitObservation { stream: "triage_parse_failure" }`
  through the dispatcher and return a no-op summary

**Recommendation:** (c) first, (a) as a follow-up. Non-destructive,
still exercises the dispatcher, gives the session log an observable
failure mode to train the flywheel on.

### 2. Confidence vocabulary

LLMs emit English more reliably than single letters. Options:

- **(a)** Raw letters `T/P/U/D/F/C` — compact but brittle
- **(b)** Words `true/probable/unknown/doubtful/false/contradictory` —
  verbose but robust; map to letters in the parser
- **(c)** Claude-native hint: few-shot example in the system prompt

**Recommendation:** (b). Parser converts to letters. Add one few-shot
example to anchor the shape.

### 3. Recursion guard

Two layers of defense:

1. System prompt explicitly forbids `ix_triage_session`
2. Parser hard-rejects any plan item with that tool name

**Decision:** both layers, non-negotiable. LLMs ignore "do not" prompts
at nontrivial rates.

### 4. ReadContext source

Current dispatch-action test uses `ReadContext::synthetic_for_legacy()`.
The triage tool should do the same until a real session-scoped
ReadContext accumulator exists. Not a blocker; a note for the Lens
refactor doc.

### 5. Flywheel ingest recursion

Calling `ix_trace_ingest` through `registry_bridge::dispatch` from
inside the triage handler means that ingest itself also runs through
the middleware chain (loop-detect + approval). That is correct — we
want the trace self-learning loop to be governed, not a sidecar. But
it means the triage session will emit a second wave of session events
from the ingest call. The caller needs to understand this when reading
the trace.

**Decision:** document it, don't special-case it. Governed recursion is
a feature.

## Estimated scope

| Component | LoC (approx) |
|---|---|
| `handlers::triage_session_with_ctx` | ~220 |
| Plan parser (strict + lenient + validation) | ~80 |
| `HexavalentDistribution::from_plan` helper | ~30 |
| Integration test using `install_session_log` | ~150 |
| Tool registration in `tools.rs` + route in `call_with_ctx` | ~30 |
| Walkthrough README / docs update | ~60 |

**Total:** ~570 LoC. One focused session. Proposed commit sequence:

1. `feat(ix-agent): plan parser + hex distribution helper for triage`
2. `feat(ix-agent): ix_triage_session — end-to-end governed harness scenario`
3. `test(ix-agent): triage_session parses plan and dispatches through full chain`
4. `docs(ix): add triage session walkthrough with real trace example`

## What this unblocks

- **Governance compliance reports** — a Demerzel consumer can issue a
  directive, the ix triage session picks it up, dispatches a governed
  plan, and returns a compliance summary built from the session log.
- **Cross-repo diagnose → dispatch (Path C)** — once the in-process
  version works, exposing `ix_dispatch_action` as an MCP tool so tars's
  `diagnose_and_remediate` can pipe structured remediations becomes a
  mechanical translation.
- **Flywheel effectiveness studies** — run the triage session N times
  with varied focus hints, measure whether ingested traces improve
  subsequent plans. First real data for the meta-learning policy.

## What this does NOT do

- Does not add new primitives. All seven are already live.
- Does not resolve the Hexavalent OR decision — it only uses the
  distribution, not the algebra.
- Does not expose any Tier 3+ destructive tools. All 50 tools are
  Tier 1/2 today, so the approval middleware will auto-approve
  everything the LLM can reach. That is safe and boring on purpose.

## Status marker

Not picked up yet. The ix session that just closed at `667880e` is the
natural venue to implement this. If that session resumes, start here.
If a different session picks it up, read this doc first and then the
three files cited in `project_context_dag_wip.md` §"Key files on resume"
(`registry_bridge.rs`, `flywheel.rs`, `session_log_wiring.rs`).
