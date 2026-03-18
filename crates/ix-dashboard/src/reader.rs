use crate::state::*;
use std::path::Path;

/// Read all JSON files from a directory matching the given extension
fn read_json_dir<T: serde::de::DeserializeOwned>(dir: &Path) -> Vec<T> {
    let mut results = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return results,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                match serde_json::from_str::<T>(&content) {
                    Ok(item) => results.push(item),
                    Err(e) => eprintln!("Warning: failed to parse {}: {}", path.display(), e),
                }
            }
        }
    }
    results
}

/// Read all belief states from state/beliefs/
pub fn read_beliefs(state_dir: &Path) -> Vec<BeliefState> {
    read_json_dir(&state_dir.join("beliefs"))
}

/// Read all evolution entries from state/evolution/
pub fn read_evolution(state_dir: &Path) -> Vec<EvolutionEntry> {
    read_json_dir(&state_dir.join("evolution"))
}

/// Read all PDCA states from state/pdca/
pub fn read_pdca(state_dir: &Path) -> Vec<PdcaState> {
    read_json_dir(&state_dir.join("pdca"))
}

/// Read all conscience signals from state/conscience/signals/
pub fn read_signals(state_dir: &Path) -> Vec<ConscienceSignal> {
    read_json_dir(&state_dir.join("conscience").join("signals"))
}

/// Load all dashboard data from a state directory
pub fn load_all(state_dir: &Path) -> DashboardData {
    DashboardData {
        beliefs: read_beliefs(state_dir),
        evolution: read_evolution(state_dir),
        pdca: read_pdca(state_dir),
        signals: read_signals(state_dir),
    }
}
