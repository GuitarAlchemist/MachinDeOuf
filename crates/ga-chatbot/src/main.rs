//! `ga-chatbot` CLI — stub MCP server and single-shot test mode.

use clap::{Parser, Subcommand};
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
    }
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
