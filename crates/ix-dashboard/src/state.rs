use serde::Deserialize;

/// Tetravalent belief value
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum TetraValue {
    #[serde(alias = "T", alias = "true")]
    True,
    #[serde(alias = "F", alias = "false")]
    False,
    #[serde(alias = "U", alias = "unknown")]
    Unknown,
    #[serde(alias = "C", alias = "contradictory")]
    Contradictory,
}

impl std::fmt::Display for TetraValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::True => write!(f, "T"),
            Self::False => write!(f, "F"),
            Self::Unknown => write!(f, "U"),
            Self::Contradictory => write!(f, "C"),
        }
    }
}

/// A belief state entry from state/beliefs/*.belief.json
#[derive(Debug, Clone, Deserialize)]
pub struct BeliefState {
    pub proposition: String,
    pub value: TetraValue,
    pub confidence: f64,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub previous_value: Option<String>,
}

/// An evolution event
#[derive(Debug, Clone, Deserialize)]
pub struct EvolutionEvent {
    #[serde(default)]
    pub event_type: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub description: String,
}

/// An evolution log entry from state/evolution/*.evolution.json
#[derive(Debug, Clone, Deserialize)]
pub struct EvolutionEntry {
    pub artifact_name: String,
    #[serde(default)]
    pub artifact_type: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub compliance_rate: Option<f64>,
    #[serde(default)]
    pub events: Vec<EvolutionEvent>,
    #[serde(default)]
    pub recommendation: Option<String>,
}

/// PDCA state from state/pdca/*.pdca.json
#[derive(Debug, Clone, Deserialize)]
pub struct PdcaState {
    pub name: String,
    pub phase: String,
    #[serde(default)]
    pub experiment: Option<bool>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub budget: Option<String>,
}

/// Conscience signal from state/conscience/signals/*.signal.json
#[derive(Debug, Clone, Deserialize)]
pub struct ConscienceSignal {
    pub signal_id: String,
    pub signal_type: String,
    pub weight: f64,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub context: Option<String>,
}

/// All loaded dashboard data
#[derive(Debug, Default)]
pub struct DashboardData {
    pub beliefs: Vec<BeliefState>,
    pub evolution: Vec<EvolutionEntry>,
    pub pdca: Vec<PdcaState>,
    pub signals: Vec<ConscienceSignal>,
}
