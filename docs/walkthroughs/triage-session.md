# `ix_triage_session` — end-to-end governed harness walkthrough

This is the first MCP tool that exercises **every** shipped harness
primitive in a single call. It turns the substrate (context DAG,
loop detector, approval middleware, session log, fuzzy distributions,
trace flywheel, MCP sampling) from a set of tested components into
an observable agent scenario.

## What it does

```text
┌─────────────────────────────────────────────────────────────┐
│                    ix_triage_session                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Read recent events from the installed SessionLog        │
│ 2. Build a compact summary of tool calls / blocks          │
│ 3. Sample the client LLM for a JSON plan with hexavalent   │
│    confidence (T/P/U/D/F/C) per item                       │
│ 4. Parse + validate (recursion guard against self-call)    │
│ 5. Build HexavalentDistribution from plan confidences      │
│ 6. Check escalation_triggered — if plan-level C > 0.3,     │
│    return escalated status WITHOUT dispatching             │
│ 7. Sort plan by Demerzel tiebreak C > U > D > P > T > F    │
│ 8. Dispatch each action through dispatch_action (full      │
│    middleware chain: loop detect + approval)               │
│ 9. Optionally export session via flywheel + invoke         │
│    ix_trace_ingest for self-learning statistics            │
│ 10. Return synthesis: plan, distribution, dispatched       │
│     outcomes, trace dir, ingest stats                      │
└─────────────────────────────────────────────────────────────┘
```

## Primitive coverage matrix

| # | Primitive | Exercised how |
|---|---|---|
| 1 | Context DAG (`ix-context`) | Indirectly — LLM may propose `ix_context_walk` targets |
| 2 | Loop detector | Duplicate proposals trip the circuit breaker; surfaces as `ActionBlocked` in the dispatched list |
| 2b | Substrate (`ix-agent-core`) | `AgentAction::InvokeTool` construction, `SessionEvent` reads |
| 3 | Approval / blast-radius | Every dispatch runs the tier classifier; Tier 1/2 auto-approves, Tier 3+ blocks |
| 4 | Session log (`ix-session`) | `current_session_log()` read, `log.events()` iteration, `log.flush()` before export |
| 5 | Fuzzy (`ix-fuzzy`) | `HexavalentDistribution` built from plan, `escalation_triggered()` plan-level gate |
| 6 | Trace flywheel | `export_session_to_trace_dir` → `ix_trace_ingest` round-trip |
| 7 | MCP sampling | `ctx.sample()` is the whole triage decision |

## Input schema

```json
{
  "focus": "optional free-text hint (e.g. 'unblock the stats investigation')",
  "max_actions": 3,
  "learn": true
}
```

- `focus` (string, optional) — passed to the LLM as a hint
- `max_actions` (integer, 1–8, default 3) — cap on plan size
- `learn` (boolean, default `true`) — if true, run the flywheel export
  + ingest after dispatch

## Prerequisites

`ix_triage_session` **requires** an installed `SessionLog`. Two ways
to set one up:

```bash
# Environment variable (read on first access of the slot)
export IX_SESSION_LOG=/path/to/session.jsonl
```

```rust
// In-process (tests, embedded harness)
use ix_agent::registry_bridge::install_session_log;
use ix_session::SessionLog;

let log = SessionLog::open("/path/to/session.jsonl").unwrap();
install_session_log(log);
```

Calling the tool without an installed log returns:

```
ix_triage_session requires an installed SessionLog. Set IX_SESSION_LOG=<path>
or call install_session_log() before dispatching this tool.
```

## Example: happy path

Input:

```json
{
  "focus": "baseline the stats investigation",
  "max_actions": 3,
  "learn": true
}
```

LLM-produced plan (returned by `ctx.sample`):

```json
[
  {
    "tool_name": "ix_stats",
    "params": {"data": [10.0, 20.0, 30.0]},
    "confidence": "T",
    "reason": "baseline re-stats with new window"
  },
  {
    "tool_name": "ix_distance",
    "params": {"a": [1.0, 2.0], "b": [4.0, 6.0], "metric": "euclidean"},
    "confidence": "probable",
    "reason": "distance check for sanity"
  }
]
```

Result:

```json
{
  "status": "dispatched",
  "focus": "baseline the stats investigation",
  "max_actions": 3,
  "events_read": 2,
  "plan": [
    {
      "tool_name": "ix_distance",
      "params": {"a": [1.0, 2.0], "b": [4.0, 6.0], "metric": "euclidean"},
      "confidence": "P",
      "reason": "distance check for sanity"
    },
    {
      "tool_name": "ix_stats",
      "params": {"data": [10.0, 20.0, 30.0]},
      "confidence": "T",
      "reason": "baseline re-stats with new window"
    }
  ],
  "distribution": {
    "T": 0.5, "P": 0.5, "U": 0.0, "D": 0.0, "F": 0.0, "C": 0.0
  },
  "escalated": false,
  "dispatched": [
    {
      "tool_name": "ix_distance",
      "ok": true,
      "confidence": "P",
      "value": {"distance": 5.0, "metric": "euclidean"},
      "reason": "distance check for sanity"
    },
    {
      "tool_name": "ix_stats",
      "ok": true,
      "confidence": "T",
      "value": {"mean": 20.0, "std": 10.0, "min": 10.0, "max": 30.0},
      "reason": "baseline re-stats with new window"
    }
  ],
  "trace_dir": "/tmp/ix-session/traces/session.json",
  "trace_ingest": {
    "ok": true,
    "stats": {
      "total_traces": 1,
      "success_count": 1,
      "failure_count": 0,
      "avg_duration_ms": 0.0
    }
  }
}
```

### Why `ix_distance` came first

The plan was sorted by `hex_priority` descending. In the Demerzel
tiebreak order `C > U > D > P > T > F`, `P` (Probable) outranks `T`
(True) — the conservative heuristic says: when the LLM is only
*probably* sure, do that first so you learn before committing to
the T-confidence actions.

## Example: escalation path

Input: a plan where the LLM flags most items as `C` (Contradictory).

```json
[
  {"tool_name": "ix_stats", "params": {}, "confidence": "C", "reason": "..."},
  {"tool_name": "ix_fft", "params": {}, "confidence": "C", "reason": "..."},
  {"tool_name": "ix_distance", "params": {}, "confidence": "contradictory", "reason": "..."}
]
```

Result: **dispatch is blocked before any action runs.**

```json
{
  "status": "escalated",
  "reason": "plan-level contradiction mass exceeds 0.3 threshold",
  "plan": [...],
  "distribution": {
    "T": 0.0, "P": 0.0, "U": 0.0, "D": 0.0, "F": 0.0, "C": 1.0
  },
  "events_read": 2
}
```

The governance gate lives in `ix_fuzzy::escalation_triggered`: when
the plan-level `C` mass exceeds `ESCALATION_THRESHOLD` (0.3), the
distribution demands human review. The triage handler honors that
by returning early.

## Example: recursion guard

Input: a malicious or buggy LLM proposes the triage tool itself.

```json
[
  {"tool_name": "ix_triage_session", "params": {}, "confidence": "T", "reason": "infinite recursion!"}
]
```

Result:

```json
{
  "status": "parse_failed",
  "error": "plan item 0: recursion guard — refusing to propose ix_triage_session",
  "raw_response": "[{\"tool_name\": \"ix_triage_session\", ...}]",
  "events_read": 2
}
```

Two layers of defense:

1. **System prompt** — explicitly tells the LLM not to propose the
   tool ("HARD CONSTRAINT: do NOT propose ix_triage_session").
2. **Parser** — `triage::parse_plan` hard-rejects any plan item whose
   `tool_name` matches `RECURSION_GUARD_TOOL`, regardless of what the
   system prompt said. Defense in depth: LLMs ignore "do not" prompts
   at non-trivial rates.

## Status fields reference

| `status` | Meaning |
|---|---|
| `dispatched` | Happy path — plan ran through the full chain |
| `escalated` | Plan-level `C` mass > 0.3 → escalated before dispatch |
| `parse_failed` | LLM response couldn't be parsed as a valid plan (malformed JSON, missing fields, recursion attempt, bad confidence label) |

## Self-learning loop (`learn: true`)

When `learn: true` (default), after dispatch the handler:

1. Calls `log.flush()` to ensure dispatched events are on disk
2. Computes `trace_dir = log.path().parent().unwrap().join("traces")`
3. Calls `flywheel::export_session_to_trace_dir(&log, &trace_dir, None)`
4. Invokes `ix_trace_ingest` via `registry_bridge::dispatch` on the
   exported directory
5. Returns the ingest stats in `trace_ingest.stats`

**Governed recursion is a feature.** The `ix_trace_ingest` call also
runs through the middleware chain, so loop detection and approval
fire on it too. A runaway triage loop that re-ingests the same trace
dozens of times will trip the circuit breaker.

## Failure modes surfaced in the output

The tool is designed to **not throw** on normal failures. Each failure
mode shows up as a field in the result:

| Failure | How it surfaces |
|---|---|
| LLM emits garbage JSON | `status: "parse_failed"` + `raw_response` |
| LLM proposes `ix_triage_session` | `status: "parse_failed"` + recursion error |
| LLM proposes bad confidence label | `status: "parse_failed"` + `BadConfidence` error |
| Plan escalation triggered | `status: "escalated"`, no dispatch |
| Individual action blocked by loop-detect | `dispatched[i].ok == false`, `dispatched[i].error` contains "circuit breaker" |
| Individual action blocked by approval | `dispatched[i].ok == false`, `dispatched[i].error` contains "ApprovalRequired" |
| Flywheel export fails | `trace_dir: null`, `trace_ingest.ok: false` |
| No session log installed | Tool returns `Err` — triage without history is meaningless |

## Integration test reference

See `crates/ix-agent/tests/triage_session.rs` for four end-to-end
tests with a stub sampling client:

- `happy_path_dispatches_plan_and_exports_trace`
- `escalation_blocks_dispatch_when_contradiction_dominates`
- `recursion_guard_surfaces_parse_failure`
- `refuses_without_installed_session_log`

The stub client demonstrates the full bidirectional JSON-RPC pattern:
it reads outbound envelopes from the `ServerContext` receiver,
extracts the `id`, and calls `ctx.deliver_response` with a canned
`sampling/createMessage` response. That pattern is the template for
any future test that needs to drive a context-aware MCP tool.

## What this unblocks

Now that one tool exercises every primitive end-to-end, several
follow-on scenarios become straightforward:

- **Compliance loops** — a Demerzel directive → triage plan →
  governed dispatch → compliance report generated from the session
  log's ActionCompleted / ActionBlocked counts.
- **Cross-repo Path C** — expose `ix_dispatch_action` as an MCP tool
  so tars's `diagnose_and_remediate` can pipe structured JSON
  remediations into the same governed chain. See
  `tars/mcp-server/src/index.ts` near `diagnose_and_remediate` for
  the pointer comment.
- **Flywheel effectiveness studies** — call triage repeatedly with
  different `focus` hints, measure whether ingested trace stats
  improve subsequent plans. First real data for the `meta-learning`
  policy in Demerzel.
- **Governance chaos testing** — run triage with deliberately
  contradictory focus hints and measure how the escalation gate
  behaves. Validates the hexavalent tiebreak in adversarial
  conditions.
