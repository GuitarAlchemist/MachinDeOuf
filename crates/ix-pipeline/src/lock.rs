//! `ix.lock` — a reproducibility manifest written next to `ix.yaml`.
//!
//! Phase 1 is **write-only**: the lock is regenerated from scratch on every
//! `ix pipeline run` to provide an audit trail. Hash verification (refusing
//! to run when `ix.lock` doesn't match current `ix.yaml` stages) arrives in
//! Phase 2.
//!
//! Format (YAML):
//!
//! ```yaml
//! schema: ix-lock/v1
//! generated: 2026-04-05T14:32:11Z
//! stages:
//!   load:
//!     skill: stats
//!     args_hash: sha256:3f2c8d9a…
//!     deps: []
//!     duration_ms: 3
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::executor::PipelineResult;
use crate::spec::PipelineSpec;

pub const LOCK_SCHEMA: &str = "ix-lock/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFile {
    pub schema: String,
    pub generated: String,
    pub stages: BTreeMap<String, LockedStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedStage {
    pub skill: String,
    pub args_hash: String,
    pub deps: Vec<String>,
    pub duration_ms: u64,
    pub cache_hit: bool,
}

impl LockFile {
    /// Build a `LockFile` from a successfully-executed pipeline.
    pub fn from_run(spec: &PipelineSpec, result: &PipelineResult) -> Self {
        let mut stages = BTreeMap::new();
        for (id, stage) in &spec.stages {
            let args_hash = hash_json(&stage.args);
            let node_result = result.node_results.get(id);
            let (duration_ms, cache_hit) = node_result
                .map(|r| (r.duration.as_millis() as u64, r.cache_hit))
                .unwrap_or((0, false));
            stages.insert(
                id.clone(),
                LockedStage {
                    skill: stage.skill.clone(),
                    args_hash,
                    deps: stage.deps.clone(),
                    duration_ms,
                    cache_hit,
                },
            );
        }
        LockFile {
            schema: LOCK_SCHEMA.into(),
            generated: current_iso_timestamp(),
            stages,
        }
    }

    /// Serialize to YAML.
    pub fn to_yaml_string(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Load a lock from disk (for future verification).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, LockError> {
        let text = std::fs::read_to_string(path)?;
        let lf: LockFile = serde_yaml::from_str(&text)?;
        if lf.schema != LOCK_SCHEMA {
            return Err(LockError::SchemaMismatch {
                expected: LOCK_SCHEMA.into(),
                actual: lf.schema,
            });
        }
        Ok(lf)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("schema mismatch: expected {expected}, got {actual}")]
    SchemaMismatch { expected: String, actual: String },
}

/// Stable content hash of a JSON `Value`. Uses the serialized form so that
/// structurally-equivalent JSON produces the same hash regardless of key
/// insertion order (BTreeMap ordering from serde_json).
fn hash_json(v: &Value) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let serialized = canonicalize(v);
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    format!("fnv1a64:{:016x}", hasher.finish())
}

fn canonicalize(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| k.as_str());
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{k:?}:{}", canonicalize(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonicalize).collect();
            format!("[{}]", parts.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn current_iso_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (year, month, day) = days_to_ymd(days);
    let h = (secs % 86_400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::StageSpec;
    use serde_json::json;

    #[test]
    fn hash_is_stable_across_key_ordering() {
        let a = json!({ "x": 1, "y": 2 });
        let b = json!({ "y": 2, "x": 1 });
        assert_eq!(hash_json(&a), hash_json(&b));
    }

    #[test]
    fn hash_differs_with_different_values() {
        let a = json!({ "x": 1 });
        let b = json!({ "x": 2 });
        assert_ne!(hash_json(&a), hash_json(&b));
    }

    #[test]
    fn lock_roundtrips_through_yaml() {
        let mut stages = BTreeMap::new();
        stages.insert(
            "load".to_string(),
            StageSpec {
                skill: "stats".into(),
                args: json!({"data": [1.0, 2.0]}),
                deps: vec![],
                cache: None,
            },
        );
        let spec = PipelineSpec {
            version: "1".into(),
            params: BTreeMap::new(),
            stages,
            x_editor: Value::Null,
        };

        // Empty PipelineResult stand-in
        let result = PipelineResult {
            node_results: Default::default(),
            total_duration: std::time::Duration::ZERO,
            cache_hits: 0,
            execution_order: vec![],
        };

        let lock = LockFile::from_run(&spec, &result);
        assert_eq!(lock.schema, LOCK_SCHEMA);
        assert_eq!(lock.stages.len(), 1);
        assert!(lock.stages["load"].args_hash.starts_with("fnv1a64:"));

        let yaml = lock.to_yaml_string().unwrap();
        let back: LockFile = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.stages["load"].skill, "stats");
        assert_eq!(back.stages["load"].args_hash, lock.stages["load"].args_hash);
    }
}
