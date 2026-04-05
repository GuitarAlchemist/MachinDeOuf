//! Bridge between `ix-registry`'s capability registry and the MCP `Tool`
//! surface exposed by `ix-agent`.
//!
//! Skills registered via `#[ix_skill]` are adapted to MCP tools by:
//!   1. Mapping the dotted skill name (`supervised.linear_regression.fit`) to
//!      an underscore MCP name (`ix_supervised_linear_regression_fit`).
//!   2. Using the skill's hand-written `json_schema` (via `schema_fn = ...`)
//!      as the MCP input-schema.
//!   3. Wrapping the handler so the MCP caller passes one JSON blob in and
//!      gets one JSON blob out.
//!
//! For composite MCP handlers we annotate a single-arg wrapper fn of shape
//! `fn(params: serde_json::Value) -> Result<serde_json::Value, String>` with
//! `#[ix_skill]`. The registry carries a single `Json → Json` socket pair,
//! but the MCP schema returned to clients is the original hand-written one.

use crate::tools::Tool;
use ix_registry::SkillDescriptor;
use ix_types::Value as IxValue;
use serde_json::Value as JsonValue;

/// Translate a dotted registry name into the MCP tool name convention.
///
/// `supervised.linear_regression.fit` → `ix_supervised_linear_regression_fit`
pub fn mcp_name(skill_name: &str) -> String {
    format!("ix_{}", skill_name.replace('.', "_"))
}

/// Dispatch a registered skill by MCP name. Wraps the JSON blob into a
/// single-element `[IxValue::Json]` slice, invokes the registry, and unwraps.
pub fn dispatch(mcp_tool_name: &str, params: JsonValue) -> Result<JsonValue, String> {
    // Undo the `ix_` prefix + underscore replacement to find the dotted name.
    let skill_name = mcp_to_skill_name(mcp_tool_name)
        .ok_or_else(|| format!("not a registry-backed tool: {mcp_tool_name}"))?;
    let descriptor = ix_registry::by_name(&skill_name)
        .ok_or_else(|| format!("skill not in registry: {skill_name}"))?;

    let args = [IxValue::Json(params)];
    let out = (descriptor.fn_ptr)(&args).map_err(|e| e.to_string())?;
    match out {
        IxValue::Json(j) => Ok(j),
        // If a skill ever returns a non-Json value from the composite-handler
        // path, surface it losslessly through serde.
        other => serde_json::to_value(other).map_err(|e| e.to_string()),
    }
}

/// Undo `mcp_name`: `ix_supervised_linear_regression_fit` → dotted skill name
/// matched against the registry by progressively collapsing underscores.
fn mcp_to_skill_name(mcp_tool_name: &str) -> Option<String> {
    let rest = mcp_tool_name.strip_prefix("ix_")?;
    // The registry stores names with dots. The MCP name replaced dots with
    // underscores, so we must scan the registry for a name whose MCP form
    // matches. O(n) on total skill count — trivial.
    for desc in ix_registry::all() {
        if mcp_name(desc.name) == mcp_tool_name {
            return Some(desc.name.to_string());
        }
    }
    // Fallback: heuristic dot-splitting (works for skills with no dots in
    // segment names, which is the convention we enforce).
    Some(rest.replace('_', "."))
}

/// Produce an MCP [`Tool`] definition for every skill in the registry. Used
/// by `ToolRegistry::register_all` to merge registry-sourced tools with any
/// remaining manual entries.
pub fn all_registry_tools() -> Vec<Tool> {
    ix_registry::all()
        .map(|desc| Tool {
            name: mcp_name_static(desc),
            description: desc.doc,
            input_schema: (desc.json_schema)(),
            handler: registry_handler_for(desc),
        })
        .collect()
}

/// Leak the MCP-name `String` to satisfy `Tool`'s `&'static str` contract.
/// The leak is bounded — one `String` per skill, once, at process startup.
fn mcp_name_static(desc: &'static SkillDescriptor) -> &'static str {
    // Interning via `OnceLock` keyed on the skill pointer identity.
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static INTERN: OnceLock<std::sync::Mutex<HashMap<&'static str, &'static str>>> =
        OnceLock::new();
    let map = INTERN.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    if let Some(&s) = guard.get(desc.name) {
        return s;
    }
    let leaked: &'static str = Box::leak(mcp_name(desc.name).into_boxed_str());
    guard.insert(desc.name, leaked);
    leaked
}

/// Per-descriptor handler: routes back through the registry by the stored
/// descriptor's name. Returned fn-pointer has static lifetime by using a
/// per-skill closure lifted into a stored `Box<dyn Fn>` via `OnceLock`.
///
/// MCP tool handlers must be `fn(Value) -> Result<Value, String>`, a plain
/// function pointer with no captures. To carry the skill name, we generate
/// one thunk per tool through dynamic dispatch stored in an interning table
/// — but since we can't lift a `Box<dyn Fn>` to `fn`, we instead use the
/// MCP name inside a generic dispatch path. `ToolRegistry::call` knows the
/// tool's name and calls [`dispatch`] directly for registry-backed tools.
fn registry_handler_for(_: &'static SkillDescriptor) -> fn(JsonValue) -> Result<JsonValue, String> {
    // Stored as a marker; `ToolRegistry::call` intercepts before invoking.
    registry_handler_marker
}

/// Sentinel handler — never executed directly. `ToolRegistry::call` sees this
/// pointer and routes to [`dispatch`] with the tool name.
pub fn registry_handler_marker(_: JsonValue) -> Result<JsonValue, String> {
    Err("registry_handler_marker should be intercepted by ToolRegistry::call".into())
}

/// True iff `handler` is the registry marker — i.e. the tool is backed by a
/// registry skill, not a manual handler.
pub fn is_registry_backed(handler: fn(JsonValue) -> Result<JsonValue, String>) -> bool {
    (handler as *const ()) == (registry_handler_marker as *const ())
}
