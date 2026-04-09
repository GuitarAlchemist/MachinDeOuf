//! Demo scenario registry.

mod chaos_detective;
mod cost_anomaly_hunter;
mod governance_gauntlet;
mod sprint_oracle;

use super::{DemoScenario, ScenarioMeta};

/// All registered demo scenarios.
pub fn all() -> Vec<Box<dyn DemoScenario>> {
    vec![
        Box::new(chaos_detective::ChaosDetective),
        Box::new(cost_anomaly_hunter::CostAnomalyHunter),
        Box::new(governance_gauntlet::GovernanceGauntlet),
        Box::new(sprint_oracle::SprintOracle),
    ]
}

/// All scenario metadata (without instantiating full scenarios).
pub fn all_meta() -> Vec<&'static ScenarioMeta> {
    vec![
        &chaos_detective::META,
        &cost_anomaly_hunter::META,
        &governance_gauntlet::META,
        &sprint_oracle::META,
    ]
}
