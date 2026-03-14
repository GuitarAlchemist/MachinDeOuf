//! Build a Data Pipeline
//!
//! Chain multiple operations as a DAG with parallel execution and memoization.
//!
//! ```bash
//! cargo run --example dag_pipeline
//! ```

use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

fn main() {
    let pipeline = PipelineBuilder::new()
        .source("raw_data", || {
            Ok(json!({"values": [1.0, 2.0, 3.0, 4.0, 5.0]}))
        })
        .node("stats", |b| {
            b.input("data", "raw_data").compute(|inputs| {
                let vals = inputs["data"]["values"].as_array().unwrap();
                let sum: f64 = vals.iter().map(|v| v.as_f64().unwrap()).sum();
                let mean = sum / vals.len() as f64;
                Ok(json!({"mean": mean, "count": vals.len()}))
            })
        })
        .node("normalize", |b| {
            b.input("data", "raw_data")
                .input("stats", "stats")
                .compute(|inputs| {
                    let vals = inputs["data"]["values"].as_array().unwrap();
                    let mean = inputs["stats"]["mean"].as_f64().unwrap();
                    let normalized: Vec<f64> =
                        vals.iter().map(|v| v.as_f64().unwrap() - mean).collect();
                    Ok(json!(normalized))
                })
        })
        .build()
        .unwrap();

    // "stats" and initial data fetch run first, then "normalize"
    let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
    println!("Normalized: {}", result.output("normalize").unwrap());
    println!("Executed in {} levels", result.execution_order.len());
}
