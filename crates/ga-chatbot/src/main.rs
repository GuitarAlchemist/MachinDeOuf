//! `ga-chatbot` CLI — stub MCP server, single-shot test mode, and QA runner.

use clap::{Parser, Subcommand};
use ga_chatbot::aggregate::{JudgeVerdict, QaResult};
use ga_chatbot::qa::{load_corpus_ids, run_deterministic_checks};
use ga_chatbot::{ask_stub, load_fixtures, ChatbotRequest, Instrument};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ga-chatbot", about = "Domain-specific voicing chatbot")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a minimal JSON-RPC stdio MCP server (stub mode).
    Serve {
        /// Use stub fixtures for responses.
        #[arg(long)]
        stub: bool,
        /// Path to stub fixtures JSONL file.
        #[arg(long, default_value = "tests/adversarial/fixtures/stub-responses.jsonl")]
        fixtures: PathBuf,
    },
    /// Single-shot ask mode for testing.
    Ask {
        /// The question to ask.
        #[arg(long)]
        question: String,
        /// Target instrument.
        #[arg(long, default_value = "guitar")]
        instrument: String,
        /// Path to stub fixtures JSONL file.
        #[arg(long, default_value = "tests/adversarial/fixtures/stub-responses.jsonl")]
        fixtures: PathBuf,
    },
    /// Run the deterministic QA pipeline on the adversarial corpus.
    Qa {
        /// Directory containing adversarial prompt corpus (*.jsonl files).
        #[arg(long)]
        corpus: PathBuf,
        /// Path to stub response fixtures (JSONL).
        #[arg(long)]
        fixtures: PathBuf,
        /// Directory containing voicing corpus JSON files.
        #[arg(long)]
        corpus_dir: PathBuf,
        /// Output path for findings (JSONL).
        #[arg(long, default_value = "findings.jsonl")]
        output: PathBuf,
    },
}

/// A prompt entry from the adversarial corpus.
#[derive(serde::Deserialize)]
struct CorpusEntry {
    id: String,
    #[allow(dead_code)]
    category: String,
    prompt: String,
    #[allow(dead_code)]
    expected_check: String,
    #[allow(dead_code)]
    expected_verdict: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { stub: _, fixtures } => {
            let fixture_map = load_fixtures(&fixtures);
            serve_jsonrpc(&fixture_map);
        }
        Commands::Ask {
            question,
            instrument,
            fixtures,
        } => {
            let fixture_map = load_fixtures(&fixtures);
            let inst = match instrument.to_lowercase().as_str() {
                "guitar" => Some(Instrument::Guitar),
                "bass" => Some(Instrument::Bass),
                "ukulele" => Some(Instrument::Ukulele),
                _ => None,
            };
            let req = ChatbotRequest {
                question,
                instrument: inst,
            };
            let resp = ask_stub(&req, &fixture_map);
            println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        }
        Commands::Qa {
            corpus,
            fixtures,
            corpus_dir,
            output,
        } => {
            std::process::exit(run_qa(&corpus, &fixtures, &corpus_dir, &output));
        }
    }
}

/// Run the deterministic QA pipeline over all adversarial prompts.
///
/// Returns 0 if no F/D verdicts, 1 otherwise.
fn run_qa(corpus_path: &std::path::Path, fixtures_path: &std::path::Path, corpus_dir: &std::path::Path, output_path: &std::path::Path) -> i32 {
    // Load fixtures for stub responses
    let fixture_map = load_fixtures(fixtures_path);

    // Load voicing corpus IDs from all *-corpus.json files in corpus_dir
    let mut all_corpus_ids = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(corpus_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && path.file_name().is_some_and(|n| n.to_string_lossy().contains("-corpus"))
            {
                let ids = load_corpus_ids(&path);
                all_corpus_ids.extend(ids);
            }
        }
    }
    eprintln!("Loaded {} voicing IDs from corpus", all_corpus_ids.len());

    // Load all adversarial prompts from corpus directory
    let prompts = load_adversarial_prompts(corpus_path);
    eprintln!("Loaded {} adversarial prompts", prompts.len());

    if prompts.is_empty() {
        eprintln!("No prompts found in {:?}", corpus_path);
        return 1;
    }

    // Run pipeline
    let mut results: Vec<QaResult> = Vec::new();
    let mut fail_count = 0;
    let mut pass_count = 0;

    for entry in &prompts {
        let req = ChatbotRequest {
            question: entry.prompt.clone(),
            instrument: Some(Instrument::Guitar),
        };
        let response = ask_stub(&req, &fixture_map);
        let findings = run_deterministic_checks(&entry.id, &entry.prompt, &response, &all_corpus_ids);

        // Determine worst verdict from deterministic checks
        let det_verdict = worst_verdict(&findings);

        // Create a single "deterministic" judge verdict for aggregation
        let judge_verdict = JudgeVerdict {
            judge: "deterministic".to_string(),
            verdict: det_verdict,
            grounded: !findings.iter().any(|f| f.layer == 1 && f.verdict == 'F'),
            accurate: true, // deterministic layer doesn't check accuracy
            safe: !findings.iter().any(|f| f.layer == 0 && f.verdict == 'F'),
            reasoning: findings
                .iter()
                .map(|f| f.reason.clone())
                .collect::<Vec<_>>()
                .join("; "),
            flags: findings
                .iter()
                .filter(|f| f.verdict == 'F')
                .map(|f| format!("layer{}:{}", f.layer, f.reason))
                .collect(),
        };

        let aggregate = ga_chatbot::aggregate::aggregate_verdicts(std::slice::from_ref(&judge_verdict));

        let result = QaResult {
            prompt_id: entry.id.clone(),
            deterministic_verdict: Some(det_verdict),
            judge_verdicts: vec![judge_verdict],
            aggregate,
        };

        match det_verdict {
            'F' | 'D' => fail_count += 1,
            _ => pass_count += 1,
        }

        results.push(result);
    }

    // Write findings to output JSONL
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out_file = match std::fs::File::create(output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file {:?}: {}", output_path, e);
            return 1;
        }
    };
    for result in &results {
        if let Ok(json) = serde_json::to_string(result) {
            writeln!(out_file, "{}", json).ok();
        }
    }

    // Print summary
    let total = results.len();
    println!();
    println!("=== Adversarial QA Summary ===");
    println!("Total prompts: {}", total);
    println!("Pass (T/P):    {}", pass_count);
    println!("Fail (F/D):    {}", fail_count);
    println!("Output:        {:?}", output_path);
    println!();

    // Print worst-scoring prompts
    let failures: Vec<_> = results
        .iter()
        .filter(|r| matches!(r.deterministic_verdict, Some('F') | Some('D')))
        .collect();
    if !failures.is_empty() {
        println!("Failing prompts:");
        for r in &failures {
            let reasons: Vec<_> = r.judge_verdicts.iter().flat_map(|j| &j.flags).collect();
            println!(
                "  {} [{}] {}",
                r.prompt_id,
                r.deterministic_verdict.unwrap_or('?'),
                reasons
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        println!();
    }

    if fail_count > 0 { 1 } else { 0 }
}

/// Find the worst verdict in a set of findings (F > D > U > C > P > T).
fn worst_verdict(findings: &[ga_chatbot::qa::Finding]) -> char {
    let priority = |c: char| -> u8 {
        match c {
            'F' => 5,
            'D' => 4,
            'C' => 3,
            'U' => 2,
            'P' => 1,
            'T' => 0,
            _ => 0,
        }
    };
    findings
        .iter()
        .map(|f| f.verdict)
        .max_by_key(|&c| priority(c))
        .unwrap_or('U')
}

/// Load all adversarial prompts from JSONL files in a directory.
fn load_adversarial_prompts(dir: &std::path::Path) -> Vec<CorpusEntry> {
    let mut prompts = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read corpus directory {:?}: {}", dir, e);
            return prompts;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "jsonl") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<CorpusEntry>(line) {
                        Ok(entry) => prompts.push(entry),
                        Err(e) => {
                            eprintln!("Failed to parse line in {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
    }
    prompts
}

/// Minimal JSON-RPC stdio loop implementing the `ga_chatbot_ask` tool.
///
/// Reads one JSON-RPC request per line from stdin, dispatches to `ask_stub`,
/// writes one JSON-RPC response per line to stdout. No async, no tokio.
fn serve_jsonrpc(fixtures: &HashMap<String, ga_chatbot::ChatbotResponse>) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                writeln!(stdout, "{}", err_resp).ok();
                stdout.flush().ok();
                continue;
            }
        };

        let id = request.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let method = request
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("");

        let response = match method {
            "ga_chatbot_ask" => {
                let params = request.get("params").cloned().unwrap_or_default();
                let question = params
                    .get("question")
                    .and_then(|q| q.as_str())
                    .unwrap_or("")
                    .to_string();
                let instrument = params
                    .get("instrument")
                    .and_then(|i| i.as_str())
                    .and_then(|i| match i {
                        "guitar" => Some(Instrument::Guitar),
                        "bass" => Some(Instrument::Bass),
                        "ukulele" => Some(Instrument::Ukulele),
                        _ => None,
                    });

                let req = ChatbotRequest {
                    question,
                    instrument,
                };
                let resp = ask_stub(&req, fixtures);
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": resp
                })
            }
            _ => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("Method not found: {}", method)
                    }
                })
            }
        };

        writeln!(stdout, "{}", response).ok();
        stdout.flush().ok();
    }
}
