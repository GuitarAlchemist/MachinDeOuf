//! `ga-chatbot` — domain-specific voicing chatbot with deterministic QA harness.
//!
//! A stub MCP server that answers grounded questions about chord voicings on
//! guitar, bass, and ukulele. Every voicing cited must resolve to a real row
//! in the corpus. The QA harness layers deterministic checks (sanitization,
//! corpus grounding, confidence thresholds) before expensive LLM judges.

pub mod aggregate;
pub mod qa;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Supported instruments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Instrument {
    Guitar,
    Bass,
    Ukulele,
}

/// A question to the chatbot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatbotRequest {
    /// Natural language question about chord voicings.
    pub question: String,
    /// Target instrument. If omitted, defaults to guitar.
    pub instrument: Option<Instrument>,
}

/// A source citation backing a voicing reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// File path to the corpus artifact.
    pub path: String,
    /// Row index in the corpus JSON array.
    pub row: usize,
}

/// The chatbot's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatbotResponse {
    /// Natural language answer grounded in corpus data.
    pub answer: String,
    /// Corpus voicing IDs cited in the answer (e.g., "guitar_v042").
    pub voicing_ids: Vec<String>,
    /// Alignment-policy confidence score (0.0-1.0).
    pub confidence: f64,
    /// File paths and row numbers backing each cited voicing.
    pub sources: Vec<Source>,
}

/// A fixture entry mapping a prompt prefix to a canned response.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureEntry {
    prompt_prefix: String,
    response: ChatbotResponse,
}

/// Load canned stub responses from a JSONL fixture file.
///
/// Each line is a JSON object with `prompt_prefix` and `response` fields.
/// Returns a map from lowercase prompt prefix to the canned response.
pub fn load_fixtures(path: &Path) -> HashMap<String, ChatbotResponse> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<FixtureEntry>(line) {
            map.insert(entry.prompt_prefix.to_lowercase(), entry.response);
        }
    }
    map
}

/// Answer a question using canned stub responses.
///
/// Looks up `req.question` by fuzzy prefix match against the fixture map.
/// If no match is found, returns a refusal with confidence 0.0.
pub fn ask_stub(req: &ChatbotRequest, fixtures: &HashMap<String, ChatbotResponse>) -> ChatbotResponse {
    let question_lower = req.question.to_lowercase();

    // Try fuzzy prefix match: find the longest fixture prefix that matches
    let mut best_match: Option<&ChatbotResponse> = None;
    let mut best_len = 0;

    for (prefix, response) in fixtures {
        if question_lower.starts_with(prefix) && prefix.len() > best_len {
            best_match = Some(response);
            best_len = prefix.len();
        }
    }

    match best_match {
        Some(response) => response.clone(),
        None => ChatbotResponse {
            answer: "I don't have enough information to answer that.".to_string(),
            voicing_ids: vec![],
            confidence: 0.0,
            sources: vec![],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_fixture_file() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"{{"prompt_prefix": "does voicing guitar_v042", "response": {{"answer": "Yes, guitar_v042 is a dyad voicing at frets x-10-8-x-x-x.", "voicing_ids": ["guitar_v042"], "confidence": 0.92, "sources": [{{"path": "state/voicings/guitar-corpus.json", "row": 42}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"prompt_prefix": "show me voicing guitar_v000", "response": {{"answer": "guitar_v000 is a dyad at frets 8-8-x-x-x-x.", "voicing_ids": ["guitar_v000"], "confidence": 0.95, "sources": [{{"path": "state/voicings/guitar-corpus.json", "row": 0}}]}}}}"#).unwrap();
        f
    }

    #[test]
    fn stub_returns_canned_response() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        let req = ChatbotRequest {
            question: "Does voicing guitar_v042 exist in the corpus?".to_string(),
            instrument: Some(Instrument::Guitar),
        };
        let resp = ask_stub(&req, &fixtures);
        assert!(resp.answer.contains("guitar_v042"));
        assert_eq!(resp.voicing_ids, vec!["guitar_v042"]);
        assert!(resp.confidence > 0.9);
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0].row, 42);
    }

    #[test]
    fn stub_refuses_unknown() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        let req = ChatbotRequest {
            question: "What is the meaning of life?".to_string(),
            instrument: None,
        };
        let resp = ask_stub(&req, &fixtures);
        assert!(resp.confidence < f64::EPSILON);
        assert!(resp.voicing_ids.is_empty());
        assert!(resp.answer.contains("don't have enough information"));
    }

    #[test]
    fn stub_parses_fixture_file() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        assert_eq!(fixtures.len(), 2);
        assert!(fixtures.contains_key("does voicing guitar_v042"));
        assert!(fixtures.contains_key("show me voicing guitar_v000"));
    }
}
