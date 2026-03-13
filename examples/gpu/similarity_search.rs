//! GPU-Accelerated Similarity Search
//!
//! Find most similar vectors using WGPU compute shaders (Vulkan/DX12/Metal).
//!
//! ```bash
//! cargo run --example similarity_search
//! ```

use machin_gpu::batch::top_k_similar;
use machin_gpu::context::GpuContext;

fn main() {
    // Initialize GPU (Vulkan on Windows/Linux, Metal on Mac)
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU available: {}", e);
            return;
        }
    };

    let query = vec![0.1, 0.5, 0.3, 0.8];
    let corpus = vec![
        vec![0.2, 0.4, 0.3, 0.7],
        vec![0.9, 0.1, 0.0, 0.2],
        vec![0.1, 0.6, 0.2, 0.9],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![0.0, 0.0, 0.1, 0.9],
    ];

    // Top-3 most similar vectors (runs on GPU)
    let results = top_k_similar(&ctx, &query, &corpus, 3);
    println!("Query: {:?}\n", query);
    println!("Top-3 most similar:");
    for (idx, score) in results {
        println!("  Vector {} {:?} -- similarity: {:.4}", idx, corpus[idx], score);
    }
}
